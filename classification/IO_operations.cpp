#include "IO_operations.hpp"

void split(const string &s, char delim, std::vector<string> &elems) {
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
}

int load_data(string file_path, data& data_samples)
{

	fstream csv_stream;
	//string file_path = "/Users/liuhengyue/Desktop/debug_log.csv";
	string entry;
	csv_stream.open(file_path,ios_base::in);
    getline(csv_stream, entry);//skip the first line
	while(getline(csv_stream, entry)){
        //split string, input idxes: 1,3,4; output idx: 9
		std::vector<string> tokens;
		split(entry, ',', tokens);
        //load input to matrices
        input_vector cur_input;
        cur_input(0) = atoi(tokens[1].c_str());
        cur_input(1) = atoi(tokens[3].c_str());
        cur_input(2) = atoi(tokens[4].c_str());
        data_samples.push_back(make_pair(cur_input, atof(tokens[9].c_str())));
	}
	return 0;
}

int load_data(string file_path, std::vector<parameter_vector>& io_vecs){
    fstream csv_stream;
    //string file_path = "/Users/liuhengyue/Desktop/debug_log.csv";
    string entry;
    csv_stream.open(file_path,ios_base::in);
    getline(csv_stream, entry);//skip the first line
    while(getline(csv_stream, entry)){
        //split string, input idxes: 1,3,4; output idx: 9
        std::vector<string> tokens;
        split(entry, ',', tokens);
        //load input to matrices
        parameter_vector cur_input;
        cur_input(0) = atof(tokens[1].c_str());
        cur_input(1) = atof(tokens[3].c_str());
        cur_input(2) = atof(tokens[4].c_str());
        cur_input(3) = atof(tokens[9].c_str());
        io_vecs.push_back(cur_input);
    }
    return 0;

}
int load_data(string file_path, std::vector<mlp_vector>& io_vecs){
    fstream csv_stream;
    //string file_path = "/Users/liuhengyue/Desktop/debug_log.csv";
    string entry;
    csv_stream.open(file_path,ios_base::in);
    getline(csv_stream, entry);//skip the first line
    while(getline(csv_stream, entry)){
        //split string, input idxes: 1,3,4; output idx: 9
        std::vector<string> tokens;
        split(entry, ',', tokens);
        //load input to matrices
        mlp_vector cur_input;
        cur_input(0) = atof(tokens[1].c_str());
        cur_input(1) = atof(tokens[3].c_str());
        cur_input(2) = atof(tokens[4].c_str());
        cur_input(3) = atof(tokens[6].c_str());
        cur_input(4) = atof(tokens[9].c_str());
        io_vecs.push_back(cur_input);
    }
    return 0;
    
}