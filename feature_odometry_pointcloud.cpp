// -----------------------------------------------------------------------------
// Homework 2 – Robot Engineering (Part 2)
// Author : Muneeb Elahi Malik
//
// Improvements (2025‑05‑04):
//   • Bottom panel now autoscales & redraws the WHOLE trajectory each frame.
//   • Adds projected point‑cloud dots (light cyan) for visual context.
//   • Uses the same resolution as the RGB image (≈1241×376 for KITTI) so
//     no quality is lost when stacking.
// -----------------------------------------------------------------------------

#include <opencv2/opencv.hpp>
#include <fstream>
#include <numeric>
#include <deque>
#include <iomanip>

using namespace std;
using namespace cv;

// ‑‑‑‑‑ user paths ‑‑‑‑‑
string image_path = "/mnt/c/Users/mm17889/Feature_tracking/first_200_right/"; // change me
// ‑‑‑‑‑ outputs ‑‑‑‑‑
const string traj_vid   = "traj_out.avi";
const string cloud_vid  = "cloud_out.avi";
const string result_vid = "HW2_sample_result_video.mp4";
const string traj_txt   = "trajectory.txt";
const string cloud_ply  = "cloud.ply";

// KITTI 00 – 02 intrinsics
const Matx33d K(7.070493e+02, 0.0,           6.040814e+02,
                0.0,           7.070493e+02, 1.805066e+02,
                0.0,           0.0,           1.0);

static inline Vec3b random_color() {
    static RNG rng(12345);
    return Vec3b(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
}

int main() {
    // params
    const int  num_frames       = 200;
    const int  max_corners      = 1500;
    const double q_level        = 0.01;
    const double min_dist       = 7.0;
    const float max_disp        = 50.f;
    const float max_err_thresh  = 15.f;

    // containers
    vector<Point3d> trajectory;           // camera centres
    vector<Point3d> cloud_world;          // accumulated cloud

    vector<Point2f> prev_pts, curr_pts;
    vector<uchar>   status; vector<float> err;
    Mat prev_gray, curr_gray, prev_frame, curr_frame;

    // video writers
    VideoWriter traj_writer, cloud_writer, result_writer;
    Size frame_sz;                        // RGB image size (≈1241×376)

    // pose (world←camera)
    Matx33d R_wc = Matx33d::eye();
    Vec3d   t_wc = Vec3d::zeros();

    // running bounds (for auto‑scaling)
    double min_x =  1e9, max_x = -1e9,
           min_z =  1e9, max_z = -1e9;

    for (int i = 0; i < num_frames; ++i) {

        // ‑‑‑‑‑ load frame ‑‑‑‑‑
        string fname = image_path + (stringstream() << setw(6) << setfill('0') << i << ".png").str();
        curr_frame   = imread(fname, IMREAD_COLOR);
        if (curr_frame.empty()) { cerr<<"Could not load "<<fname<<'\n'; break; }
        cvtColor(curr_frame, curr_gray, COLOR_BGR2GRAY);

        // ‑‑‑‑‑ first frame: just corner detect & setup writers ‑‑‑‑‑
        if (i == 0) {
            goodFeaturesToTrack(curr_gray, curr_pts, max_corners, q_level, min_dist);
            frame_sz = curr_frame.size();                 // e.g. 1241×376

            traj_writer .open(traj_vid,  VideoWriter::fourcc('M','J','P','G'), 20, Size(800,800), true);
            cloud_writer.open(cloud_vid, VideoWriter::fourcc('M','J','P','G'), 20, Size(800,800), true);
            result_writer.open(result_vid,
                               VideoWriter::fourcc('a','v','c','1'),
                               20,
                               Size(frame_sz.width, frame_sz.height*2), // RGB + map
                               true);

            if (!traj_writer.isOpened() || !cloud_writer.isOpened() || !result_writer.isOpened()) {
                cerr<<"✖ could not open output videos\n"; return -1;
            }

            trajectory.emplace_back(t_wc);
            prev_pts  = curr_pts;
            prev_gray = curr_gray.clone();
            prev_frame= curr_frame.clone();
            continue;
        }

        // ‑‑‑‑‑ LK tracking ‑‑‑‑‑
        TermCriteria term(TermCriteria::COUNT|TermCriteria::EPS, 30, 0.01);
        calcOpticalFlowPyrLK(prev_gray, curr_gray,
                             prev_pts, curr_pts,
                             status, err, Size(21,21), 3, term);

        vector<Point2f> p_prev, p_curr;
        for (size_t k=0;k<curr_pts.size();++k) {
            if (!status[k]) continue;
            float d = norm(prev_pts[k]-curr_pts[k]);
            if (d < max_disp && err[k] < max_err_thresh) {
                p_prev.push_back(prev_pts[k]);
                p_curr.push_back(curr_pts[k]);
            }
        }
        if (p_prev.size() < 8) { cerr<<"✖ too few matches\n"; break; }

        // ‑‑‑‑‑ pose from essential ‑‑‑‑‑
        Mat inliers;
        Mat E = findEssentialMat(p_prev, p_curr, K, RANSAC, 0.999, 1.0, inliers);
        Mat R_ct, t_ct;
        recoverPose(E, p_prev, p_curr, K, R_ct, t_ct, inliers);

        R_wc = R_wc * Matx33d(R_ct);
        t_wc = t_wc + R_wc * Vec3d(t_ct);
        trajectory.emplace_back(t_wc);

        // update bounds
        min_x = min(min_x, t_wc[0]);  max_x = max(max_x, t_wc[0]);
        min_z = min(min_z, t_wc[2]);  max_z = max(max_z, t_wc[2]);

        // ‑‑‑‑‑ triangulate & extend cloud ‑‑‑‑‑
        Matx34d P0 = K * Matx34d::eye();
        Matx34d P1; hconcat(R_ct, t_ct, P1); P1 = K * P1;

        Mat pts4D; triangulatePoints(P0,P1,p_prev,p_curr,pts4D);
        for (int c=0;c<pts4D.cols;++c) {
            Mat col = pts4D.col(c); col/=col.at<float>(3);
            Vec3d pt(col.at<float>(0),col.at<float>(1),col.at<float>(2));
            if (abs(pt[2])<0.1||abs(pt[2])>1e3) continue;
            cloud_world.emplace_back(R_wc*pt + t_wc); // bring to world frame
        }

        // -------------- MAP CANVAS (same size as RGB) -------------------------
        Mat map(frame_sz.height, frame_sz.width, CV_8UC3, Scalar::all(0));

        // scale factors
        double range_x = max(max_x-min_x, 1e-6);
        double range_z = max(max_z-min_z, 1e-6);
        double sx = (map.cols-60)/range_x, sz = (map.rows-60)/range_z;
        double sc = min(sx, sz);

        auto projectXZ = [&](double X, double Z)->Point{
            int u = int(30 + (X - min_x)*sc);
            int v = int(map.rows - (30 + (Z - min_z)*sc));
            return Point(u,v);
        };

        // draw projected cloud (light‑cyan)
        for (const auto& p : cloud_world) {
            Point pix = projectXZ(p.x, p.z);
            if (pix.inside(Rect(0,0,map.cols,map.rows)))
                map.at<Vec3b>(pix) = Vec3b(200,255,255);
        }

        // draw trajectory as thick cyan polyline
        for (size_t k=1;k<trajectory.size();++k){
            Point a = projectXZ(trajectory[k-1].x, trajectory[k-1].z);
            Point b = projectXZ(trajectory[k  ].x, trajectory[k  ].z);
            line(map,a,b,Scalar(255,255,0),3,LINE_AA);
        }
        // current pose marker (red)
        circle(map, projectXZ(t_wc[0],t_wc[2]), 5, Scalar(0,0,255), -1);

        putText(map,"Frame "+to_string(i),Point(10,25),
                FONT_HERSHEY_SIMPLEX,0.8,Scalar::all(255),2);

        // keep the older 800×800 debug videos (optional)
        traj_writer.write(map);  // reuse map (already nicer)

        // -------------- CLOUD ROTATION CANVAS (unchanged) ---------------------
        Mat cloud_view(800,800,CV_8UC3,Scalar::all(0));
        double theta = i*CV_PI/180;
        Matx33d R_y(cos(theta),0,sin(theta), 0,1,0, -sin(theta),0,cos(theta));
        for (const auto& p: cloud_world){
            Vec3d pr = R_y*p;
            double Z = pr[2]+5; if (Z<=0) continue;
            Point pix(int(400+pr[0]*50/Z), int(400-pr[1]*50/Z));
            if (pix.inside(Rect(0,0,800,800)))
                cloud_view.at<Vec3b>(pix)=Vec3b(255,255,255);
        }
        cloud_writer.write(cloud_view);

        // -------------- feature overlay --------------------------------------
        Mat feat = curr_frame.clone();
        for (size_t k=0;k<p_curr.size();++k){
            if (!inliers.empty() && !inliers.at<uchar>((int)k)) continue;
            circle(feat,p_curr[k],3,Scalar(0,255,0),-1);
        }

        // -------------- stack & write ----------------------------------------
        Mat composite;
        vconcat(feat, map, composite);
        result_writer.write(composite);

        // prep next iter
        goodFeaturesToTrack(curr_gray, curr_pts, max_corners, q_level, min_dist);
        prev_pts   = curr_pts;
        prev_gray  = curr_gray.clone();
    }

    // ‑‑‑‑‑ finish ‑‑‑‑‑
    traj_writer.release();
    cloud_writer.release();
    result_writer.release();

    // txt
    ofstream ft(traj_txt);
    for (auto& c: trajectory) ft<<c.x<<' '<<c.y<<' '<<c.z<<'\n';
    ft.close();

    // ply
    ofstream ply(cloud_ply);
    ply<<"ply\nformat ascii 1.0\nelement vertex "<<cloud_world.size()<<"\n"
       <<"property float x\nproperty float y\nproperty float z\n"
       <<"property uchar red\nproperty uchar green\nproperty uchar blue\n"
       <<"end_header\n";
    for (auto& p: cloud_world){
        auto col=random_color();
        ply<<p.x<<' '<<p.y<<' '<<p.z<<' '
           <<int(col[2])<<' '<<int(col[1])<<' '<<int(col[0])<<'\n';
    }
    ply.close();

    cout<<"✔ Finished.\n"
        <<"   Split‑screen video : "<<result_vid<<'\n';
    return 0;
}
