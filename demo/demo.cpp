/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdlib>
// DBoW2
#include "DBoW2.h" // defines Surf64Vocabulary and Surf64Database

#include <DUtils/DUtils.h>
#include <DVision/DVision.h>
//#include <DUtilsCV/DUtilsCV.h>
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
using namespace DBoW2;
using namespace DUtils;
using namespace std;
using namespace cv;

// - - - - - - - - - -system parameters- - - - - - - - - - - - - - -

unsigned long RESERVE = 0;//number of images to build vocabulary
unsigned long NIMAGE = 0;//number of images in the train folder
int HESSIAN = 400;
int K = 10;
int L = 7;
size_t NUM_DESCRIPTORS = 0;
// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void loadFeatures(vector<vector<vector<float> > > &features, vector<cv::String>& reference, cv::String trainFolder, unsigned long lb = 0, unsigned long ub = RESERVE);
void loadAll(vector<vector<vector<float> > > &features, vector<cv::String>& reference, cv::String trainFolder);

void changeStructure(const vector<float> &plain, vector<vector<float> > &out,
                     int L);
void testVocCreation(const vector<vector<vector<float> > > &features, vector<cv::String>& reference);
void testDatabase(const vector<vector<vector<float> > > &features, vector<cv::String> test_reference, string repo);
inline bool file_exit (const std::string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}
string toString(string str1, string str2);
unsigned long check_files_num(string folder){
    unsigned long count = 0;
    DIR *dp;
    struct dirent *dirp;
    string filepath;
    struct stat filestat;
    dp = opendir(folder.c_str());
    if (dp == NULL){
        cout << "Error (" << errno << "): Unable to open " << folder << endl;
        return -1;
    }
    
    while ((dirp = readdir(dp))) {
        filepath = folder + "/" + dirp->d_name;
        
        if (stat(filepath.c_str(), &filestat)) continue;
        if(dirp->d_name[0] == '.') continue;
        
        count++;
    }
    closedir(dp);
    return count;
}
int SURF_test_case(string path);






// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//file paths
string TRAINFOLDER = "/Users/henryliu/Downloads/out";
string TRAINFOLDER2 = "/Users/henryliu/Downloads/images";
string TESTFOLDER = "/Users/henryliu/Downloads/testing_dataset";
string STOREPATH = "/Users/henryliu/Documents/BoW_matching-test_automation/";


string DBEXT = "db.yml.gz";
string LABELEXT = "labels.yml";
string VOCEXT = "voc.yml.gz";
string LOG = "debug_log.csv";
int db_index = 0;
// ----------------------------------------------------------------------------


int main()
{

    string training_set[] = {TRAINFOLDER, TRAINFOLDER2};
    //reserve space for features
    RESERVE = 10;

    //test image
    vector<vector<vector<float> > > test_features;
    vector<cv::String> test_references;
    //extract test image features

    vector<vector<vector<float> > > features;
    vector<cv::String> reference;


 //Extract train features and build vocabulary -- trainning part
    loadFeatures(test_features, test_references, TESTFOLDER);
//    //vocabularies creation
//    //2 databases
   for(int i = 0; i < 2; i++){
       

        //check if the database exits
        if(!file_exit(toString(STOREPATH, DBEXT))){
            try {
                loadFeatures(features, reference, training_set[i], 0, 100);
                testVocCreation(features, reference);

            } catch (std::bad_alloc& ba) {
                //catch exception when the voc is too big
                std::cerr << "bad_alloc caught: " << ba.what() << '\n';
                break;
            }

        }
        //-- testing part

        //Retreive results
        testDatabase(test_features, test_references, training_set[i]);
        
        cout<<"Done"<<endl;

        //clean up
        features.clear();
        reference.clear();
        db_index++;
    }
    return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<vector<float> > > &features, vector<cv::String>& reference, cv::String trainFolder, unsigned long lb, unsigned long ub)
{

    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(HESSIAN, 4, 2, EXTENDED_SURF);

    
    cout << "Extracting SURF features..." << endl;
    DIR *dp;
    struct dirent *dirp;
    struct stat filestat;
    cv::String filepath;
    //vector<cv::String> broken;
    
    
    //Asumming query images are in a folder
    dp = opendir(trainFolder.c_str());
    
    if (dp == NULL){
        cout << "Error (" << errno << "): Unable to open " << trainFolder << endl;
        return;
    }
    int counter = 0;
    while ((dirp = readdir(dp)) && counter < ub) {
        filepath = trainFolder + "/" + dirp->d_name;
        
        if (stat(filepath.c_str(), &filestat)) continue;
        if(dirp->d_name[0] == '.') continue;
        
        //cout << float(counter+1)*100/RESERVE << "%" << endl;
        cv::Mat image = cv::imread(filepath, CV_8U); //Use this if query images are passed directly
        if(image.data==NULL) {
            //broken.push_back(dirp->d_name);
            continue;
        }

        if(counter > (int)lb - 1){
            reference.push_back(dirp->d_name);
            cv::Mat mask;
            vector<cv::KeyPoint> keypoints;
            vector<float> descriptors;
            surf->detectAndCompute(image, mask, keypoints, descriptors);
            if(descriptors.empty()) continue;
            features.push_back(vector<vector<float> >());
            changeStructure(descriptors, features.back(), surf->descriptorSize());
            NUM_DESCRIPTORS += descriptors.size();
        }
        counter++;
    }
    //change the capacity to actual size
    features.shrink_to_fit();
    //NIMAGE = (int)features.size();
    //
    //    cv::FileStorage fs("broken_images.yml", cv::FileStorage::WRITE);
    //    fs << "Image File" << broken;
    //    fs.release();
}


void changeStructure(const vector<float> &plain, vector<vector<float> > &out,
                     int L)
{
    out.resize(plain.size() / L);
    
    unsigned int j = 0;
    for(unsigned int i = 0; i < plain.size(); i += L, ++j)
    {
        out[j].resize(L);
        std::copy(plain.begin() + i, plain.begin() + i + L, out[j].begin());
    }
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<vector<vector<float> > > &features, vector<cv::String>& reference)
{
    // branching factor and depth levels
//    const int k = 15; //original 9
//    const int L = 5;  // 3
    const WeightingType weight = TF_IDF;
    const ScoringType score = L2_NORM;
    
    Surf64Vocabulary voc(K, L, weight, score);
    
    cout << "Creating a " << K << "^" << L << " vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;
    
    cout << "Vocabulary information: " << endl
    << voc << endl << endl;
    
    // lets do something with this vocabulary
    //  cout << "Matching images against themselves (0 low, 1 high): " << endl;
    //  BowVector v1, v2;
    //  for(int i = 0; i < NIMAGES; i++)
    //  {
    //    voc.transform(features[i], v1);
    //    for(int j = 0; j < NIMAGES; j++)
    //    {
    //      voc.transform(features[j], v2);
    //
    //      double score = voc.score(v1, v2);
    //      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    //    }
    //  }
    
    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    cv::FileStorage fs(toString(STOREPATH, LABELEXT), cv::FileStorage::WRITE);
    fs << "Labels" << reference;
    fs.release();
    voc.save(toString(STOREPATH, VOCEXT));
    cout << "Done" << endl;
    //create database from voc
    cout << "Creating a small database..." << endl;
    
    Surf64Database db(voc, false, 0); // false = do not use direct index
    // (so ignore the last param)
    // The direct index is useful if we want to retrieve the features that
    // belong to some vocabulary node.
    // db creates a copy of the vocabulary, we may get rid of "voc" now
    
    //add images to the database
    for(int i = 0; i < (int)features.size(); i++)
    {
        db.add(features[i]);
    }
    
    cout << "... done!" << endl;
    
    cout << "Saving database... " << endl;
    
    db.save(toString(STOREPATH, DBEXT));
    cout << "...done!" << endl;
}


void testDatabase(const vector<vector<vector<float> > > &features, vector<cv::String> references, string repo)
{
//    cout << "Creating a small database..." << endl;
//    
//    // load the vocabulary from disk
//    Surf64Vocabulary voc("/Users/henryliu/Documents/DBoW2-master/demo/kohls_voc.yml.gz");
//    
//    Surf64Database db(voc, false, 0); // false = do not use direct index
//    // (so ignore the last param)
//    // The direct index is useful if we want to retrieve the features that
//    // belong to some vocabulary node.
//    // db creates a copy of the vocabulary, we may get rid of "voc" now
//    
//    //add images to the database
//    for(int i = 0; i < nimages; i++)
//    {
//        db.add(features[i]);
//    }
//    
//    cout << "... done!" << endl;
    
    //    cout << "Retrieving database..." << endl;
    //    Surf64Database db("kohls_db.yml.gz");
    //    cout << "..done!" << endl;
    //save log
    fstream logfs;
    if(!file_exit(STOREPATH+LOG)){
        logfs.open((STOREPATH+LOG), ios_base::out | ios_base::app);
        logfs << "# of images, # of descriptors, Hessian, branch, depth, accuracy, mean, smallest, median, largest"<<endl;
    }
    else logfs.open((STOREPATH+LOG), ios_base::out | ios_base::app);
    
    //load database
    Surf64Database db;
    vector<double> scores;
    db.load(toString(STOREPATH, DBEXT));
    vector<cv::String> reference;
    cv::FileStorage fs(toString(STOREPATH, LABELEXT), cv::FileStorage::READ);
    fs["Labels"] >> reference;
    cout << "... done! This is: " << endl << db << endl;
    
    // and query the database
    cout << "Querying the database: " << endl;
    
    QueryResults ret;
    int match = 0;
    for(int i = 0; i < (int)features.size(); i++)
    {
        db.query(features[i], ret, 5); // retrieves 5 top matches
        
        
        // ret[0] is the best match, ret[0].Score gives confidence index of best match image
        
        //cout << "Searching for Image " << i << ". " << ret << endl;
        if (ret.empty()){
            cout << "No image found" << endl;
            //logfs << RESERVE << ", " << HESSIAN << ", " << K << ", " << L << ", " << "No image" << ", " << 0 << ", " << "No image" << ", " << 0 << ", " << 0 << endl;
            
        }
        else
        {
            cout << "-----------------------------------------------------"<<endl;
            cout << "For this image: " << references[i] << endl;
            cout << "Best matched Image: " << reference[ret[0].Id] << " with a score of "<< ret[0].Score << endl;
            cout << "More matches: " ;
            cout << reference[ret[1].Id]<< ", ";
            cout << reference[ret[2].Id] << ", ";
            cout << reference[ret[3].Id] << ", ";
            cout << reference[ret[4].Id] << endl;
            cout << "-----------------------------------------------------"<<endl;
            std::stringstream ss;
            ss <<repo<< "/" << reference[ret[0].Id];
            cv::Mat res_img = cv::imread(ss.str());
            
            std::stringstream ss2;
            ss2 <<TESTFOLDER<< "/" << references[i];

            cv::imshow("1st rank image", res_img);
            cvWaitKey();
            cv::Mat test_img = cv::imread(ss2.str());
            cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(HESSIAN, 4, 2, EXTENDED_SURF);
            vector<cv::Point2f> pt1, pt2;
            vector<cv::KeyPoint> keypoints, keypoints2;
            //-- Step 2: Calculate descriptors (feature vectors)

            
            Mat descriptors, descriptors2;
            surf->detectAndCompute( res_img, Mat(), keypoints, descriptors );
            surf->detectAndCompute( test_img, Mat(), keypoints2, descriptors2 );
            FlannBasedMatcher matcher;
            std::vector< DMatch > matches;
            matcher.match( descriptors, descriptors2, matches);
            for( int i = 0; i < matches.size(); i++ )
            {
                //-- Get the keypoints from the good matches
                pt1.push_back( keypoints[ matches[i].queryIdx ].pt );
                pt2.push_back( keypoints2[ matches[i].trainIdx ].pt );
            }
            cv::Mat H = cv::findHomography(pt1, pt2, CV_RANSAC);

            cout << H << endl;
            match++;
            scores.push_back(ret[0].Score);
        }

    }
//    cout << "Saving database... " << endl;
//    
//    db.save("/Users/henryliu/Documents/DBoW2-master/demo/kohls_db.yml.gz");
//    cout << "...done!" << endl;
    float acurracy = float(match)/features.size();
    //cout << "Acurracy: " << acurracy << endl;
    //store results
    sort(scores.begin(), scores.end());
    double mean = accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
    double median = scores[scores.size() / 2];
    double smallest = scores[0];
    double largest = scores[scores.size() - 1];
    //nth_element(scores.begin(), scores.begin()+scores.size()/2, scores.end());//median score
    
    logfs << RESERVE << ", " << NUM_DESCRIPTORS  << ", " << HESSIAN << ", " << K << ", " << L <<", "<< acurracy << ", "<< mean << ", "<< smallest << ", "<< median << ", "<< largest << endl;
    logfs.close();
    
}

string toString(string str1, string str2){
    std::stringstream ss;
    ss <<str1<< db_index << "_"<<"kohls"<<"_"<<str2;
    return ss.str();
}



