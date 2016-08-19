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


using namespace DBoW2;
using namespace DUtils;
using namespace std;


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//system parameters
unsigned long RESERVE = 0;//number of images to build vocabulary
unsigned long NIMAGE = 0;//number of images in the train folder
int HESSIAN = 400;
int K = 8;
int L = 3;
size_t NUM_DESCRIPTORS = 0;
// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;

void loadFeatures(vector<vector<vector<float> > > &features, vector<cv::String>& reference, cv::String trainFolder, unsigned long num = RESERVE);
void loadAll(vector<vector<vector<float> > > &features, vector<cv::String>& reference, cv::String trainFolder);

void changeStructure(const vector<float> &plain, vector<vector<float> > &out,
                     int L);
void testVocCreation(const vector<vector<vector<float> > > &features, vector<cv::String>& reference);
void testDatabase(const vector<vector<vector<float> > > &features, vector<cv::String> test_reference);
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
string TESTFOLDER = "/Users/henryliu/Downloads/test_images2";
string STOREPATH = "/Users/henryliu/Documents/DBoW2-master/demo/";

//string STOREPATH = "/Users/henry/Downloads/results/";

string DBEXT = "kohls_db.yml.gz";
string LABELEXT = "kohls_labels.yml";
string VOCEXT = "kohls_voc.yml.gz";
string LOG = "debug_log.csv";
// ----------------------------------------------------------------------------
//test case paths
string IMAGEPATH = "/Users/henryliu/Downloads/cbir_test_folder/41772.3.jpg";
string TESTSUMMARY = "/Users/henryliu/Documents/DBoW2-master/demo/hessian-#_of_interest_points-data.csv";

int main()
{
//    SURF_test_case(IMAGEPATH);

    //reserve space for features
    RESERVE = check_files_num(TRAINFOLDER);

    //test image
    vector<vector<vector<float> > > features2;
    vector<cv::String> reference2;
    //extract test image features
        //NIMAGE = check_files_num(TRAINFOLDER);
    vector<vector<vector<float> > > features;
    vector<cv::String> reference;
    //for(RESERVE = 20; RESERVE < 571; RESERVE +=50){

    for(RESERVE = 2500; RESERVE < 2501; RESERVE +=50){
 //Extract train features and build vocabulary -- trainning part
//        loadFeatures(features, reference, TRAINFOLDER);
//        for(K = 17; K < 21; K += 2){
//            for (L = 2; L < 14; L ++){
        loadFeatures(features, reference, TRAINFOLDER, 2040);
        loadFeatures(features, reference, TRAINFOLDER2);
        loadFeatures(features2, reference2, TESTFOLDER);
        //
//        for(K = 11; K < 21; K += 2){
//            for (L = 3; L < 14; L ++){
        for(K = 19; K < 20; K += 2){
            for (L = 6; L < 7; L ++){

                //check if the database exits
                if(!file_exit(toString(STOREPATH, DBEXT))){
                    try {

                        testVocCreation(features, reference);

                    } catch (std::bad_alloc& ba) {
                        //catch exception when the voc is too big
                        std::cerr << "bad_alloc caught: " << ba.what() << '\n';
                        break;
                    }

                }
                //-- testing part

                //Retreive results
                testDatabase(features2, reference2);
                
                cout<<"Done"<<endl;

            }
        }
        NUM_DESCRIPTORS = 0;
        //clean up
        features.clear();
        reference.clear();
    }
    return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<vector<float> > > &features, vector<cv::String>& reference, cv::String trainFolder, unsigned long num)
{

    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(HESSIAN, 4, 2, EXTENDED_SURF);

    
    cout << "Extracting SURF features..." << endl;
    DIR *dp;
    struct dirent *dirp;
    struct stat filestat;
    cv::String filepath;
    vector<cv::String> broken;
    
    
    //Asumming query images are in a folder
    dp = opendir(trainFolder.c_str());
    
    if (dp == NULL){
        cout << "Error (" << errno << "): Unable to open " << trainFolder << endl;
        return;
    }
    int counter = 0;
    while ((dirp = readdir(dp)) && counter < num) {
        filepath = trainFolder + "/" + dirp->d_name;
        
        if (stat(filepath.c_str(), &filestat)) continue;
        if(dirp->d_name[0] == '.') continue;
        
        //cout << float(counter+1)*100/RESERVE << "%" << endl;
        cv::Mat image = cv::imread(filepath, CV_8U); //Use this if query images are passed directly
        if(image.data==NULL) {
            broken.push_back(dirp->d_name);
            continue;
        }

        reference.push_back(dirp->d_name);
        cv::Mat mask;
        vector<cv::KeyPoint> keypoints;
        vector<float> descriptors;
        surf->detectAndCompute(image, mask, keypoints, descriptors);
        if(descriptors.empty()) continue;
        features.push_back(vector<vector<float> >());
        changeStructure(descriptors, features.back(), surf->descriptorSize());
        NUM_DESCRIPTORS += descriptors.size();
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

//void loadAll(vector<vector<vector<float> > > &features, vector<cv::String>& reference, cv::String trainFolder)
//{
//    features.clear();
//    //    features.reserve(RESERVE);
//    features.reserve(RESERVE);
//    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(HESSIAN, 4, 2, EXTENDED_SURF);
//    
//    
//    cout << "Extracting SURF features..." << endl;
//    DIR *dp;
//    struct dirent *dirp;
//    struct stat filestat;
//    cv::String filepath;
//    vector<cv::String> broken;
//    
//    
//    //Asumming query images are in a folder
//    dp = opendir(trainFolder.c_str());
//    
//    if (dp == NULL){
//        cout << "Error (" << errno << "): Unable to open " << trainFolder << endl;
//        return;
//    }
//    unsigned long count = 0;
//    while ((dirp = readdir(dp)) && count < RESERVE) {
//        filepath = trainFolder + "/" + dirp->d_name;
//        
//        if (stat(filepath.c_str(), &filestat)) continue;
//        if(dirp->d_name[0] == '.') continue;
//        
//        cout << filepath << endl;
//        cv::Mat image = cv::imread(filepath, CV_8U); //Use this if query images are passed directly
//        if(image.data==NULL) {
//            broken.push_back(dirp->d_name);
//            continue;
//        }
//
//        reference.push_back(dirp->d_name);
//        cv::Mat mask;
//        vector<cv::KeyPoint> keypoints;
//        vector<float> descriptors;
//        surf->detectAndCompute(image, mask, keypoints, descriptors);
//        if(descriptors.empty()) continue;
//        features.push_back(vector<vector<float> >());
//        changeStructure(descriptors, features.back(), surf->descriptorSize());
//        NUM_DESCRIPTORS += descriptors.size();
//        count++;
//        //cout<<count<<endl;
//    }
//    //change the capacity to actual size
//    features.shrink_to_fit();
//    //NIMAGE = (int)features.size();
//    //
//    //    cv::FileStorage fs("broken_images.yml", cv::FileStorage::WRITE);
//    //    fs << "Image File" << broken;
//    //    fs.release();
//}
// ----------------------------------------------------------------------------

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


void testDatabase(const vector<vector<vector<float> > > &features, vector<cv::String> test_reference)
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
            //cout << "No image found" << endl;
            //logfs << RESERVE << ", " << HESSIAN << ", " << K << ", " << L << ", " << "No image" << ", " << 0 << ", " << "No image" << ", " << 0 << ", " << 0 << endl;
            
        }
        else if(reference[ret[0].Id] == test_reference[i])
        {
            //cout << "Best matched Image: " << reference[ret[0].Id] << " with a score of "<< ret[0].Score << endl;
//            cout << "More matches: " << endl;
//           cout << reference[ret[1].Id]<< endl;
//            cout << reference[ret[2].Id] << endl;
//            cout << reference[ret[3].Id] << endl;
//            cout << reference[ret[4].Id] << endl;
//            cout<<RESERVE << ", " << HESSIAN << ", " << K << ", " << L << ", " << reference[ret[0].Id] << ", " << ret[0].Score << ", " << reference[ret[1].Id] << ", " << ret[1].Score << endl;
            match++;
            scores.push_back(ret[0].Score);
        }
        else{
            //cout<<"wrong match"<<endl;
        }
    }
//    cout << "Saving database... " << endl;
//    
//    db.save("/Users/henryliu/Documents/DBoW2-master/demo/kohls_db.yml.gz");
//    cout << "...done!" << endl;
    float acurracy = float(match)/features.size();
    cout << "Acurracy: " << acurracy << endl;
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
    ss <<str1<<RESERVE<<"_"<< HESSIAN <<"_"<<K<<"_"<<L<<"_"<<str2;
    return ss.str();
}


/**
 * Test the relationship between Hessian threshold and number of interest points
 */
//int SURF_test_case(string path)
//{
//    cout<<"Start!...";
//
//    cv::Mat img = imread( path, cv::IMREAD_GRAYSCALE );
//
//    
//    if( !img.data)
//    { std::cout<< " --(!) Error reading images " << std::endl; return -1; }
//    
//    //-- Step 1: Detect the keypoints using SURF Detector
//    //int minHessian = 400;
//    
//    fstream fs;
//    fs.open(TESTSUMMARY, ios_base::out | ios_base::trunc);
//    fs << "Hessian," <<"#ofKeyPoints"<<endl;
//    
//    for(int hessianT =1500; hessianT<1501; hessianT++){
//        cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(hessianT, 4, 2, EXTENDED_SURF);
//        std::vector<cv::KeyPoint> keypoints;
//        vector<float> descriptors;
//        surf->detectAndCompute(img, cv::Mat(), keypoints, descriptors);
//        
//
//        fs << hessianT << "," << (int)keypoints.size() << ","<<endl;
//        
//        //-- Draw keypoints
//        cv::Mat img_keypoints;
//        
//        drawKeypoints( img, keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
//
//        //-- Show detected (drawn) keypoints
//        imshow("Keypoints", img_keypoints );
//
//        cv::waitKey(0);
////        return (int)keypoints.size();
//    }
//    cout<<"Done!"<<endl;
//    fs.close();
//    return 0;
//}
