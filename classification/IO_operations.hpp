//
//  IO_operations.hpp
//  Solve_least_squre
//
//  Created by LiuHengyue on 8/2/16.
//  Copyright © 2016 LiuHengyue. All rights reserved.
//

#ifndef IO_operations_h
#define IO_operations_h
#include <dlib/optimization.h>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>

using namespace std;
using namespace dlib;
typedef matrix<int,3,1> input_vector;
typedef matrix<double,4,1> parameter_vector;
typedef matrix<double,5,1> mlp_vector;

typedef std::vector<std::pair<input_vector, double> > data;

void split(const string &s, char delim, std::vector<string> &elems);
int load_data(string file_path, data& data_samples);
int load_data(string file_path, std::vector<parameter_vector>& io_vecs);
int load_data(string file_path, std::vector<mlp_vector>& io_vecs);

#endif /* IO_operations_h */
