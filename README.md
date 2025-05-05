## Features Used
Shi-Tomasi corner detection for identifying good initial features to track.

Lucas-Kanade Optical Flow (Pyramidal) for tracking features between consecutive frames.

## Algorithm
For the first frame, detect Shi-Tomasi corners using goodFeaturesToTrack.

For each new frame, track features from the previous frame using calcOpticalFlowPyrLK.

Filter matches based on displacement and optical flow error thresholds.

Draw lines and circles to visualize tracked features and motion vectors.

Replenish lost features if the count falls below a threshold.

Annotate and save visualizations as single-frame and dual-view videos.

Log tracking statistics (mean error, tracked/lost/added features) to a CSV file.