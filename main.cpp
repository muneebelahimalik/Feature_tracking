// Assignment 1
// Muneeb Elahi Malik
// Intro to Robotics

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <fstream>
#include <deque>

using namespace std;
using namespace cv;

// WSL-compatible image directory (Since I used WSL for this assignment) 
string image_path = "/mnt/c/Users/mm17889/Feature_tracking/first_200_right/";   // Change this to your image directory

int main() {
    const int num_frames = 200;              // Number of frames to process
    const int max_corners = 1000;            // Max corners to track
    const double quality_level = 0.01;       // Quality level for corner detection
    const double min_distance = 7.0;         // Minimum distance between corners
    const float max_displacement = 50.0;     // Max displacement for tracking
    const float max_error_threshold = 15.0;  // Max error threshold for tracking
    const int smoothing_window = 5;          // Smoothing window for error calculation

    Size frame_size;                          
    VideoWriter writer_single, writer_dual;

    vector<Point2f> prev_pts, curr_pts; 
    vector<uchar> status; 
    vector<float> err; 
    vector<float> avg_errors; 
    vector<int> tracked_counts, lost_counts, added_counts; 
    deque<float> error_buffer; 

    Mat prev_gray, curr_gray, prev_frame, curr_frame; 

    for (int i = 0; i < num_frames; ++i) { 
        stringstream ss; 
        ss << setw(6) << setfill('0') << i << ".png";  // Change to file extension if needed
        string filename = image_path + ss.str(); 

        curr_frame = imread(filename, IMREAD_COLOR); 
        if (curr_frame.empty()) { 
            cerr << "Could not read image: " << filename << endl; 
            break;
        }

        cvtColor(curr_frame, curr_gray, COLOR_BGR2GRAY); 
        if (i == 0) {
            // Detect initial corners using Shi-Tomasi
            goodFeaturesToTrack(curr_gray, curr_pts, max_corners, quality_level, min_distance); 
            frame_size = curr_frame.size(); 

            writer_single.open("feature_tracking_single.avi", VideoWriter::fourcc('M','J','P','G'), 20, frame_size, true); 
            writer_dual.open("feature_tracking_dual.avi", VideoWriter::fourcc('M','J','P','G'), 20, Size(frame_size.width * 2, frame_size.height), true); 

            if (!writer_single.isOpened() || !writer_dual.isOpened()) {
                cerr << "Error: Could not open output video files." << endl; 
                return -1;
            }
        } else {
            // Optical flow calculation using Lucas-Kanade method
            TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, 20, 0.03); 
            calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, curr_pts, status, err, Size(21, 21), 3, criteria);

            Mat display_single = curr_frame.clone(); 
            Mat display_dual;
            hconcat(prev_frame, curr_frame, display_dual);          // Concatenate frames for dual view

            float total_error = 0.0; 
            int num_valid_errors = 0; 

            for (size_t j = 0; j < curr_pts.size(); ++j) {
                float displacement = norm(prev_pts[j] - curr_pts[j]);
                // Filter based on error and displacement
                if (status[j] && displacement < max_displacement && err[j] < max_error_threshold) {
                    Scalar color = displacement > 20 ? Scalar(0, 0, 255) : Scalar(0, 255, 0);
                    // Draw tracked features and motion
                    circle(display_single, curr_pts[j], 2, color, -1);
                    line(display_single, prev_pts[j], curr_pts[j], Scalar(0, 255, 255));

                    Point2f pt_prev = prev_pts[j];
                    Point2f pt_curr = curr_pts[j] + Point2f(prev_frame.cols, 0);
                    circle(display_dual, pt_prev, 2, color, -1);
                    circle(display_dual, pt_curr, 2, Scalar(0, 0, 255), -1);
                    line(display_dual, pt_prev, pt_curr, Scalar(255, 0, 0), 1);

                    total_error += err[j];
                    ++num_valid_errors;
                }
            }
            // Calculate mean error and smooth it
            float mean_error = num_valid_errors > 0 ? total_error / num_valid_errors : 0.0;
            error_buffer.push_back(mean_error);
            if (error_buffer.size() > smoothing_window) error_buffer.pop_front();

            float smoothed_error = accumulate(error_buffer.begin(), error_buffer.end(), 0.0f) / error_buffer.size();
            avg_errors.push_back(smoothed_error);
            // Annotate frames with tracking information
            putText(display_single, "Frame: " + to_string(i), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
            putText(display_dual, "Frame: " + to_string(i), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

            writer_single.write(display_single);
            writer_dual.write(display_dual);

            imshow("Single Frame Tracking", display_single);
            imshow("Dual View Tracking", display_dual);   
            if (waitKey(1) == 27) break;
            // Filter and keep good points
            vector<Point2f> filtered_pts;
            for (size_t j = 0; j < curr_pts.size(); ++j) {
                if (status[j] && norm(prev_pts[j] - curr_pts[j]) < max_displacement && err[j] < max_error_threshold) {
                    filtered_pts.push_back(curr_pts[j]);
                }
            }
            // Replenish features if count drops below threshold
            int num_tracked = (int)filtered_pts.size();
            int num_lost = (int)(prev_pts.size() - num_tracked);
            int num_added = 0;

            if (num_tracked < 500) {
                vector<Point2f> new_pts;
                goodFeaturesToTrack(curr_gray, new_pts, max_corners, quality_level, min_distance);
                num_added = (int)new_pts.size();
                filtered_pts.insert(filtered_pts.end(), new_pts.begin(), new_pts.end());
            }
            // Store frame statistics
            tracked_counts.push_back(num_tracked);
            lost_counts.push_back(num_lost);
            added_counts.push_back(num_added);

            cout << "Frame " << i
                 << " | Tracked: " << num_tracked
                 << " | Lost: " << num_lost
                 << " | Added: " << num_added
                 << " | Mean Error (smoothed): " << smoothed_error
                 << endl;

            curr_pts = filtered_pts;
        }

        prev_pts = curr_pts;
        prev_gray = curr_gray.clone();
        prev_frame = curr_frame.clone();
    }
    // Write statistics to CSV for later analysis
    ofstream file("tracking_stats.csv");
    file << "Frame,MeanError,Tracked,Lost,Added\n";
    for (size_t i = 0; i < avg_errors.size(); ++i) {
        file << i << "," << avg_errors[i] << ","
             << tracked_counts[i] << ","
             << lost_counts[i] << ","
             << added_counts[i] << "\n";
    }
    file.close();

    float total_avg_error = accumulate(avg_errors.begin(), avg_errors.end(), 0.0f) / avg_errors.size();
    cout << "\nTracking Complete." << endl;
    cout << "Avg. Smoothed Error Across Frames: " << total_avg_error << endl;
    cout << "Videos: 'feature_tracking_single.avi' + 'feature_tracking_dual.avi'\n CSV: 'tracking_stats.csv'\n";

    writer_single.release();
    writer_dual.release();
    destroyAllWindows();
    return 0;
}
