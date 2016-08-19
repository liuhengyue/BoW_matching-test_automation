//
//  TestImageGenerator.cpp
//  BoW2-learning
//
//  Created by Henry on 8/4/16.
//  Copyright © 2016 LiuHengyue. All rights reserved.
//

#include "TestImageGenerator.hpp"

void loadImages(cv::String folder);
void modifyImage(Mat& input, Mat& output);
string path = "/Users/henryliu/Downloads/out";
string write_path = "/Users/henryliu/Downloads/test_images2/";
//int main( int argc, char** argv )
//{
////    Mat input = imread(path);
////    imshow("test image", input);
////    waitKey();
//    loadImages(path);
//    return 0;
//    
//}

void loadImages(cv::String folder)
{
    DIR *dp;
    struct dirent *dirp;
    struct stat filestat;
    cv::String filepath;
    //Asumming query images are in a folder
    dp = opendir(folder.c_str());
    if (dp == NULL){
        cout << "Error (" << errno << "): Unable to open " << folder << endl;
        return;
    }
    int counter = 1;
    while ((dirp = readdir(dp))) {
        filepath = folder + "/" + dirp->d_name;
        
        if (stat(filepath.c_str(), &filestat)) continue;
        if(dirp->d_name[0] == '.') continue;
        
        cout << "#" << counter <<  " " << dirp->d_name << endl;
        cv::Mat image = cv::imread(filepath); //Use this if query images are passed directly
        if(image.data==NULL) continue;//some files are broken
        //do image processing here
        Mat artificial;
        modifyImage(image, artificial);
        //output image here
        imwrite(write_path + dirp->d_name, artificial);
        //imshow(dirp->d_name, artificial);
        //waitKey();
        counter++;
    }
}

void modifyImage(Mat& input, Mat& output){
    Mat intermediate_output;
    //blur
    blur(input, intermediate_output, Size(5,5));
    //noise
    Mat guassian_noise(intermediate_output.size(), intermediate_output.type());
    randu(guassian_noise, 80, 50);
    intermediate_output += guassian_noise;
    //skew
    Point2f srcTri[3];
    Point2f dstTri[3];
    /// Set the dst image the same type and size as src
    output = Mat::zeros( intermediate_output.rows, intermediate_output.cols, intermediate_output.type() );
    
    /// Set your 3 points to calculate the  Affine Transform
    srcTri[0] = Point2f( 0,0 );
    srcTri[1] = Point2f( intermediate_output.cols - 1, 0 );
    srcTri[2] = Point2f( 0, intermediate_output.rows - 1 );
    
    dstTri[0] = Point2f( intermediate_output.cols*0.0, intermediate_output.rows*0.1 );
    dstTri[1] = Point2f( intermediate_output.cols*0.85, intermediate_output.rows*0.05 );
    dstTri[2] = Point2f( intermediate_output.cols*0.05, intermediate_output.rows*0.9 );
    
    /// Get the Affine Transform
    Mat warp_mat = getAffineTransform( srcTri, dstTri );
    
    /// Apply the Affine Transform just found to the src image
    warpAffine( intermediate_output, output, warp_mat, output.size());
    output = output(Rect(output.cols*0.1, output.rows*0.1, output.cols*0.8, output.rows*0.8));

}


