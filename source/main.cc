// MIT License
//
// Copyright (c) 2018 Rene Jean Corneille
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "main.hh"

/** trains a collection of data on a collection of models **

@param data: the dictionary of data with keys the type of data and items the opencv container where data is stored

@param model: machine learning models stores within a dictionary with keys the algorithm name and items are the instances of the
machine learning

 **/

 double train_test(std::pair<std::string,std::map<std::string, cv::Mat> > data, std::pair<std::string,cv::Ptr<cv::ml::StatModel> > model, std::string path){

     cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(data.second.at("features"), 0, data.second.at("labels"));

     auto y_predict = cv::OutputArray(data.second.at("labels"));

     y_predict.clear();

     trainData->setTrainTestSplitRatio(0.2,true);

     model.second->train(trainData);

     auto error = model.second->calcError(trainData, true, y_predict);

     auto filename = path + data.first + "_" +model.first + ".txt";

     cv::Mat feature_row = cv::Mat::ones(1, 2, CV_32F);

     feature_row = data.second.at("features");

     cv::Mat label_row = cv::Mat();

     model.second->predict(feature_row,y_predict);

     std::ofstream output(filename);

     output << "x1" << "," << "x2" << "," << "y" << "\n";

     for(int i = 0; i < feature_row.rows; i++){

         auto input = cv::InputArray(feature_row.row(i));

         output << feature_row.at<float>(i,0) << "," << feature_row.at<float>(i,1)  << "," <<  y_predict.getMatRef().at<float>(i,0) << "\n";
     }

     return error;
 }

 /*
  */
 std::map<std::string, cv::Mat> readTextFile(std::string path){

     std::map<std::string, cv::Mat> result;

     std::pair<std::string, cv::Mat> features_pair,labels_pair;

     cv::Mat feature_row = cv::Mat::ones(1, 2, CV_32F);

     cv::Mat label_row = cv::Mat::ones(1, 1, CV_32S);

     cv::Mat features, labels;

     std::string str;

     std::ifstream file(path);

     float x,y,z;

     while(getline(file, str, '\n')){

         std::stringstream ss(str);

         ss >> x >> y >> z;

         std::cout << x << "\n";

         feature_row.at<float>(0, 0) = x;

         feature_row.at<float>(0, 1) = y;

         label_row.at<float>(0, 0) = z;

         features.push_back(feature_row);

         labels.push_back(label_row);
     }

     features_pair = std::pair<std::string, cv::Mat>(std::string("features"),features);
     labels_pair = std::pair<std::string, cv::Mat>(std::string("labels"),labels);

     result.insert(features_pair);
     result.insert(labels_pair);

     return result;
 }

 /*
  */
 std::unordered_map<std::string,cv::Ptr<cv::ml::StatModel> > createModels(){

     auto models = std::unordered_map<std::string,cv::Ptr<cv::ml::StatModel> >();

     auto decision_tree = cv::ml::DTrees::create();
     auto random_forest = cv::ml::RTrees::create();
     auto boost = cv::ml::Boost::create();
     auto k_nearest = cv::ml::KNearest::create();
     auto linear_svm = cv::ml::SVM::create();
     auto rbf_svm = cv::ml::SVM::create();
     auto sigmoid_svm = cv::ml::SVM::create();

     auto criter = cv::TermCriteria();

     criter.type = CV_TERMCRIT_EPS + CV_TERMCRIT_ITER;
     criter.epsilon = 1e-8;
     criter.maxCount = 5000;

     auto criter_svm = cv::TermCriteria();

     criter_svm.type = CV_TERMCRIT_EPS;
     criter_svm.epsilon = 1e-10;

     // parameters for decision tree
     decision_tree->setMaxCategories(2);
     decision_tree->setMaxDepth(3000);
     decision_tree->setMinSampleCount(1);
     decision_tree->setTruncatePrunedTree(false);
     decision_tree->setUse1SERule(false);
     decision_tree->setUseSurrogates(false);
     decision_tree->setPriors(cv::Mat());
     decision_tree->setCVFolds(1);

     models.insert(std::pair<std::string,cv::Ptr<cv::ml::StatModel> >("decision_tree",decision_tree));

     // parameters for random forest
     random_forest->setMaxCategories(2);
     random_forest->setMaxDepth(3000);
     random_forest->setMinSampleCount(1);
     random_forest->setTruncatePrunedTree(false);
     random_forest->setUse1SERule(false);
     random_forest->setUseSurrogates(false);
     random_forest->setPriors(cv::Mat());
     random_forest->setTermCriteria(criter);
     random_forest->setCVFolds(1);

     models.insert(std::pair<std::string,cv::Ptr<cv::ml::StatModel> >("random_forest",random_forest));

     // parameters for boost tree
     boost->setBoostType(cv::ml::Boost::DISCRETE);
     boost->setWeakCount(100);
     boost->setWeightTrimRate(0.95);
     boost->setMaxDepth(2000);
     boost->setUseSurrogates(false);
     boost->setPriors(cv::Mat());

     models.insert(std::pair<std::string,cv::Ptr<cv::ml::StatModel> >("boost",boost));

     // parameters for k nearest neighbors
     k_nearest->setDefaultK(5);
     k_nearest->setIsClassifier(true);

     models.insert(std::pair<std::string,cv::Ptr<cv::ml::StatModel> >("k_nearest_neighbors",k_nearest));

     // parameters for linear support vector machines
     linear_svm->setC(100);
     linear_svm->setKernel(linear_svm->LINEAR);
     linear_svm->setTermCriteria(criter_svm);
     linear_svm->setType(linear_svm->C_SVC);

     models.insert(std::pair<std::string,cv::Ptr<cv::ml::StatModel> >("linear_svm",linear_svm));

     // parameters for rbf support vector machines
     rbf_svm->setC(100);
     rbf_svm->setTermCriteria(criter);
     rbf_svm->setCoef0(0.3);
     rbf_svm->setKernel(rbf_svm->RBF);
     rbf_svm->setGamma(0.9);
     rbf_svm->setType(rbf_svm->C_SVC);

     models.insert(std::pair<std::string,cv::Ptr<cv::ml::StatModel> >("rbf_svm",rbf_svm));

     // parameters for rbf support vector machines
     sigmoid_svm->setC(100);
     sigmoid_svm->setTermCriteria(criter);
     sigmoid_svm->setCoef0(0.3);
     sigmoid_svm->setKernel(rbf_svm->SIGMOID);
     sigmoid_svm->setGamma(0.1);
     sigmoid_svm->setType(sigmoid_svm->C_SVC);

     models.insert(std::pair<std::string,cv::Ptr<cv::ml::StatModel> >("sigmoid_svm",rbf_svm));

     return models;
 }

 /*
  */
 std::unordered_map<std::string,double> run(std::pair<std::string,std::map<std::string, cv::Mat> > data,
                                            std::unordered_map<std::string,cv::Ptr<cv::ml::StatModel> > models, std::string path){
     auto errors = std::unordered_map<std::string,double>();

     for (std::pair<std::string,cv::Ptr<cv::ml::StatModel> > model:models){
         errors.insert(std::pair<std::string,double>(model.first,train_test(data,model,path)));
     }

     return errors;
 }

 /*
  */
 std::unordered_map<std::string,std::map<std::string, cv::Mat> > getData(std::string path){

     auto res = std::unordered_map<std::string,std::map<std::string, cv::Mat> >();

     // type 1 small sample, low noise

     auto type1_small_low = readTextFile(path + "moon_small_low.txt");

     res.insert(std::pair<std::string,std::map<std::string, cv::Mat> >("moon,small,low",type1_small_low));

     // type 1 large sample, low noise

     auto type1_large_low = readTextFile(path + "moon_large_low.txt");

     res.insert(std::pair<std::string,std::map<std::string, cv::Mat> >("moon,large,low",type1_large_low));

     // type 1 large sample, high noise

     auto type1_large_high = readTextFile(path + "moon_large_high.txt");

     res.insert(std::pair<std::string,std::map<std::string, cv::Mat> >("moon,large,high",type1_large_high));

     // type 1 small sample, high noise

     auto type1_small_high = readTextFile(path + "moon_small_high.txt");

     res.insert(std::pair<std::string,std::map<std::string, cv::Mat> >("moon,small,high",type1_small_high));

     // type 2 small sample, low noise

     auto type2_small_low = readTextFile(path + "circle_small_low.txt");

     res.insert(std::pair<std::string,std::map<std::string, cv::Mat> >("circle,small,low",type2_small_low));

     // type 2 large sample, low noise

     auto type2_large_low = readTextFile(path + "circle_large_low.txt");

     res.insert(std::pair<std::string,std::map<std::string, cv::Mat> >("circle,large,low",type2_large_low));

     // type 2 large sample, high noise

     auto type2_large_high = readTextFile(path + "circle_large_high.txt");

     res.insert(std::pair<std::string,std::map<std::string, cv::Mat> >("circle,large,high",type2_large_high));

     // type 2 small sample, high noise

     auto type2_small_high = readTextFile(path + "circle_small_high.txt");

     res.insert(std::pair<std::string,std::map<std::string, cv::Mat> >("circle,small,high",type2_small_high));

     // type 3 small sample, low noise

     auto type3_small_low = readTextFile(path + "blob_small_low.txt");

     res.insert(std::pair<std::string,std::map<std::string, cv::Mat> >("blob,small,low",type3_small_low));

     // type 3 large sample, low noise

     auto type3_large_low = readTextFile(path + "blob_large_low.txt");

     res.insert(std::pair<std::string,std::map<std::string, cv::Mat> >("blob,large,low",type3_large_low));

     // type 3 large sample, high noise

     auto type3_large_high = readTextFile(path + "blob_large_high.txt");

     res.insert(std::pair<std::string,std::map<std::string, cv::Mat> >("blob,large,high",type3_large_high));

     // type 3 small sample, high noise

     auto type3_small_high = readTextFile(path + "blob_small_high.txt");

     res.insert(std::pair<std::string,std::map<std::string, cv::Mat> >("blob,small,high",type3_small_high));

     return res;
 }

 // ====================================================================== //

 int main(int argc, const char * argv[]) {

     auto results = std::map<std::string,std::unordered_map<std::string,double> >();

     for (std::pair<std::string,std::map<std::string, cv::Mat> > pair: getData(std::string(argv[1]) )){
         results.insert(std::pair<std::string,std::unordered_map<std::string, double> >(pair.first,run(pair,createModels(),std::string(argv[1]))));
     }

     auto filename = std::string(argv[1]) + "all_results.txt";

     std::ofstream output(filename);

     output << "data" << "," << "size" << "," << "noise" << "," << "algo" << "," << "error" << "\n";

     for (std::map<std::string,std::unordered_map<std::string, double> >::iterator it = results.begin() ; it != results.end() ; it++){
         for (std::pair<std::string,double> p: it->second){
             output << it->first << "," << p.first  << "," << p.second << "\n";
         }
     }

     return 0;
 }
