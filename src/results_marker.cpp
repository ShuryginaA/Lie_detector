// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cstdio>
#include <iostream>

#define _USE_MATH_DEFINES
#include <cmath>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include "results_marker.hpp"
#include "face_inference_results.hpp"
#include "utils.hpp"

namespace gaze_estimation {
ResultsMarker::ResultsMarker(bool showFaceBoundingBox,
                             bool showHeadPoseAxes,
                             bool showLandmarks,
                             bool showGaze):
                             showFaceBoundingBox(showFaceBoundingBox),
                             showHeadPoseAxes(showHeadPoseAxes),
                             showLandmarks(showLandmarks),
                             showGaze(showGaze) {
}
    std::vector<cv::Point>dir_x(600);
    std::vector<cv::Point>dir_y(600);
    int c = 1, step = 1;
    cv::Mat back = cv::imread("./bground.png");
    cv::Mat img;
    
    void ResultsMarker::mark(cv::Mat& image,
        const FaceInferenceResults& faceInferenceResults) const {

        auto faceBoundingBox = faceInferenceResults.faceBoundingBox;
        auto faceBoundingBoxWidth = faceBoundingBox.width;
        auto faceBoundingBoxHeight = faceBoundingBox.height;
        auto scale = 0.002 * faceBoundingBoxWidth;
        cv::Point tl = faceBoundingBox.tl();
        img = back;
        cv::putText(img, //target image
            "Right direction", //text
            cv::Point(500, 60),
            cv::FONT_HERSHEY_DUPLEX,
            1.0,
            CV_RGB(255, 0, 0),
            2);
        cv::putText(img, //target image
            "Gaze up", //text
            cv::Point(500, 100),
            cv::FONT_HERSHEY_DUPLEX,
            1.0,
            CV_RGB(0, 0, 255),
            2);
        cv::putText(img, //target image
            "Gaze up right", //text
            cv::Point(500, 140),
            cv::FONT_HERSHEY_DUPLEX,
            1.0,
            CV_RGB(33, 0, 66),
            2);
        cv::putText(img, //target image
            "Left direction", //text
            cv::Point(500, 400 ),
            cv::FONT_HERSHEY_DUPLEX,
            1.0,
            CV_RGB(255, 0, 0),
            2);
        cv::putText(img, //target image
            "Gaze down", //text
            cv::Point(500, 440),
            cv::FONT_HERSHEY_DUPLEX,
            1.0,
            CV_RGB(0, 0, 255),
            2);
        cv::putText(img, //target image
            "Gaze down left", //text
            cv::Point(500, 480),
            cv::FONT_HERSHEY_DUPLEX,
            1.0,
            CV_RGB(33, 0, 66),
            2);

        if (showFaceBoundingBox) {
            cv::rectangle(image, faceInferenceResults.faceBoundingBox, cv::Scalar::all(255), 1);
            cv::putText(image,
                cv::format("Detector confidence: %0.2f",
                    static_cast<double>(faceInferenceResults.faceDetectionConfidence)),
                cv::Point(static_cast<int>(tl.x),
                    static_cast<int>(tl.y - 5. * faceBoundingBoxWidth / 200.)),
                cv::FONT_HERSHEY_COMPLEX, scale, cv::Scalar::all(255), 1);
        }


        if (showGaze) {
            auto gazeVector = faceInferenceResults.gazeVector;

            cv::Point2f gazeAngles;
            cv::Point3f in = gazeVector;
            cv::Point3f out;
            float yaw = faceInferenceResults.headPoseAngles.x;
            out.x = in.x * cos(yaw*M_PI/180.0) + in.z * sin(yaw*M_PI/180.0);
            out.y = in.y;
            out.z = -in.x * sin(yaw*M_PI/180.0) + in.z * cos(yaw*M_PI/180.0);
            gazeVectorToGazeAngles(out, gazeAngles);
           

            if (dir_x.size()<500 && dir_y.size() < 500)
            {
                if (abs(gazeAngles.x) > 16)
                {
                    dir_x.push_back(cv::Point(step + 15, 5 * (gazeAngles.x) + img.rows / 2));
                    step++;
                }
               
                if (abs(gazeAngles.y) > 8)
                {
                    dir_y.push_back(cv::Point(step + 15, 5 * (gazeAngles.y) + img.rows / 2));
                    step++;
                }              
            }

            else
            {
                back = cv::imread("./bground.png");
                dir_x.clear();
                step=0;
                if (abs(gazeAngles.x) > 16)
                {
                    dir_x.push_back(cv::Point(step + 15, 5 * (gazeAngles.x) + img.rows / 2));
                }
                dir_y.clear();
                if (abs(gazeAngles.y) > 8)
                {
                    dir_y.push_back(cv::Point(step + 15, 5 * (gazeAngles.y) + img.rows / 2));
                }
            }
            std::cout << dir_x.size() << "  " << dir_y.size()<<std::endl;
            img = back;
            polylines(img, dir_x, false, cv::Scalar(0, 0, 255), 2);
            polylines(img, dir_y, false, cv::Scalar(255, 0, 0), 2);
            cv::namedWindow("Graphics", 0);
            cv::resizeWindow("Graphics", 600, 500);
            cv::imshow("Graphics", img);

            cv::Point p1(0, img.rows/2), p2(1000, img.rows / 2), p3(2, 0), p4(2, 1000);
            cv::Scalar colorLine(0, 0, 0);
            int thicknessLine = 2;
            cv::line(img, p1, p2, colorLine, thicknessLine);
            cv::line(img, p3, p4, colorLine, thicknessLine);
        }
    }

    void ResultsMarker::save() {
        cv::imwrite("Graphic.jpg", img);
    }



void ResultsMarker::toggle(char key) {
    if (key == 'l') {
        showLandmarks = !showLandmarks;
    } else if (key == 'h') {
        showHeadPoseAxes = !showHeadPoseAxes;
    } else if (key == 'g') {
        showGaze = !showGaze;
    } else if (key == 'd') {
        showFaceBoundingBox = !showFaceBoundingBox;
    } else if (key == 'a') {
        showFaceBoundingBox = true;
        showHeadPoseAxes = true;
        showLandmarks = true;
        showGaze = true;
    } else if (key == 'n') {
        showFaceBoundingBox = false;
        showHeadPoseAxes = false;
        showLandmarks = false;
        showGaze = false;
    }
}
}  // namespace gaze_estimation
