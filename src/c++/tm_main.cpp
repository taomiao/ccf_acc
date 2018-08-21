#include<cstdlib>
#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<sys/time.h>
#include"tm_cpu_lib.h"
//#include"polaris/include/polaris.h"

using namespace std;

int * test_data;
int * test_feat;
long long batch_size = 20000;
float * embeddings = NULL;
//float * embeddings2 = NULL;

float * load_param_by_name(char * filename,int size){
	//cout<<"loading "<<filename<<endl;
        float * data = (float*)malloc(size*sizeof(float));
	string file_url("./params2/");
	file_url += filename;
        ifstream param_file(file_url.c_str());
        //cout<<"ok"<<endl;
	string line;
        int cnt=0;
        float * p = data;
        while(getline(param_file,line)){
                istringstream sin(line);
                float temp;
                string field;
                while(getline(sin,field,',')){
                        float tmp;
                        stringstream ss(field);
                        ss>>tmp;
                        *p=tmp;
                        p++;
                        cnt++;
                }

        }
        //cout<<cnt<<endl;
	return data;
}

int load_test_data(string test_name){
	ifstream infd(test_name.c_str());
	string line;
	int cnt=0;
	int * p = test_data;
	while(getline(infd,line)){
		istringstream sin(line);
		//float temp;
		string field;
		while(getline(sin,field,',')){
			int tmp=0;
			if (field.size() == 0)
				continue;
			stringstream ss(field);
			ss>>tmp;
			*p=tmp;
			p++;
			cnt++;
		}
	}
	//cout<<cnt<<endl;
	return 0;
}

void load_whole_data(){
	load_test_data("test_b_5000.csv");
}

int load_test_feat(string test_name){
        ifstream infd(test_name.c_str());
        string line;
        int cnt=0;
        int * p = test_feat;
        while(getline(infd,line)){
                istringstream sin(line);
                float temp;
                string field;
                while(getline(sin,field,',')){
                        int tmp=0;
                        stringstream ss(field);
                        ss>>tmp;
                        *p=tmp;
                        p++;
                        cnt++;
                }
        }
        //cout<<cnt<<endl;
        return 0;
}

void embedding_lookup(int * d_input,float * embedding_data){
	float * embd = NULL;
	embd = embeddings;
//	float *	embd2 = embeddings2;
	float * result = embedding_data;
	long long res_idx=0;
	for(long long k=0;k<batch_size;k++){
		for(long long i=0;i<40;i++){
			int idx = d_input[k*39+i];
			if(i==39)
				idx=12460;
			float * start_pos = embd+1200*idx;
			for(int j=0;j<1200;j++){
				result[res_idx]=*start_pos;
				res_idx++;
				start_pos += 1;	
			}
		}
		//for(int i=0;i<20;i++){
		//	int idx = f_input[k*20+i];
		//	float * start_pos = embd2+1800*idx;
                //        for(int j=0;j<1800;j++){
                //                result[res_idx]=*start_pos;
                //                res_idx++;
                //                start_pos += 1;
                //        }
		//}
	}
}

void trans_and_mul(float * input,float * output){
	long long idx = 0;
	for (long long i=0;i<batch_size;i++){
		for (long long m = 0;m<40;m++){
			for(long long n=0;n<40;n++){
				if(m>=n){
					continue;
				}
				for( long long k=0;k<30;k++){
					float tmp1 =(*(input+(i*40*40*30+n*40*30+m*30+k)));
					float tmp2 =(*(input+(i*40*40*30+m*40*30+n*30+k)));
					output[idx] = tmp1*tmp2;
					idx ++;
				}
			}
		}
	}
}

int main(){
	//cout<<"hello world"<<endl;
	test_data = (int*)malloc(1000000*39*sizeof(int));
	//test_feat = (int*)malloc(batch_size*20*sizeof(int));
		
	//cout<<"------------------ok"<<endl;
	embeddings = load_param_by_name("feature_embeddings.csv",12461*1200);
	//embeddings2 = load_param_by_name("feature_embeddings2.csv",10000*1800);
	
	//cout<<"------------------ok"<<endl;
	//load_test_data("./TEST_csv/TEST_00000");
	//load_test_feat("./TEST_feat/TEST_00000");
	//cout<<"------------------ok"<<endl;
	//float * test_data_embd = embedding_lookup(test_data,test_feat); //1024*60*1800
#if 0
	for (int i = 0; i < 100; i++) {
		cout << test_data_embd[i] << endl;
	}

	cout << endl;
	size_t sz = 1024*60*1800;
	for (size_t i = sz-10; i < sz; i++) {
		cout << test_data_embd[i] << endl;
	}
#endif

	//float * trans_mul_data = (float *)malloc(1024*30*59*30*sizeof(float));
	//trans_and_mul(test_data_embd,trans_mul_data);
	//float * trans_mul_data = load_param_by_name("split_mid_feature_0.csv",1024*53100);
	//cout<<"mid_file start -----------------"<<endl;
	//ofstream fout("first_line.csv");
	//for(int i=0;i<53100;i++){
	//	fout<<trans_mul_data[i]<<',';
	//}
	//cout<<"mid_file end  -----------------"<<endl;

	float * layer0 = load_param_by_name("layer_0_t.csv",23400*300);
	//cout<<"layer0_param-----------------------"<<endl;
	//for (int i=0;i<10;i++){
	//	cout<<layer0[53100*300-1-i]<<endl;
	//}
	float * bias0 = load_param_by_name("bias_0.csv",300);
	float * layer1 = load_param_by_name("layer_1_t.csv",300*300);
	float * bias1 = load_param_by_name("bias_1.csv",300);
	float * layer2 = load_param_by_name("layer_2_t.csv",300*300);
	float * bias2 = load_param_by_name("bias_2.csv",300);
	float * concat_prj = load_param_by_name("concat_projection.csv",300);
	float * concat_bias = load_param_by_name("concat_bias.csv",1);

		
	float * embedding_data = (float*)malloc(batch_size*40*1200*sizeof(float));
	float * trans_mul_data = (float *)malloc(batch_size*20*39*30*sizeof(float));
	float * layer0_out = (float*)malloc(batch_size*300*sizeof(float));
	float * layer0_a_out = (float*)malloc(batch_size*300*sizeof(float));
	float * layer1_out = (float*)malloc(batch_size*300*sizeof(float));
	float * layer1_a_out = (float*)malloc(batch_size*300*sizeof(float));
        float * layer2_out = (float*)malloc(batch_size*300*sizeof(float));
	float * layer2_a_out = (float*)malloc(batch_size*300*sizeof(float));
	float * concat_out = (float*)malloc(batch_size*300*sizeof(float));
	float * concat_a_out = (float*)malloc(batch_size*sizeof(float));

	float * result = (float *)malloc(batch_size*sizeof(float));
	
	cout<<"loading origin data ..."<<endl;
	load_whole_data();
	cout<<"finish loading data."<<endl;

	//cout<<"concat_bias"<<concat_bias[0]<<endl;
	ofstream fres("cpu_result.txt");
	ofstream time_record("cpu_time_record.csv");
	clock_t total_time = 0;
	for( int i=0;i<1000000/batch_size;i++){
		//char num[5];
		//sprintf(num,"%05d",i);
		//string s(num);	
		//load_test_data("./TEST_B_csv/TEST_"+s);
		//load_test_feat("./TEST_feat/TEST_"+s);
		int * batch_test_data = test_data + (batch_size*39*i);
		cout<<"pre-handling data ..."<<endl;
		
		timeval ycl_start,io_start,p_start,p_stop,io_stop;
		
		gettimeofday(&ycl_start,NULL);
		embedding_lookup(batch_test_data,embedding_data); //1024*60*1800
		trans_and_mul(embedding_data,trans_mul_data);
		cout<<"finish pre-handling data."<<endl;

		cpu_gemm(batch_size,300,23400,trans_mul_data,layer0,layer0_out,bias0);
		gettimeofday(&io_start,NULL);
		gettimeofday(&p_start,NULL);
		//cout<<"layer0"<<endl;
		//for (int i=0;i<10;i++){
		//	cout<<layer0_out[i]<<endl;
		//}
		cpu_activation(0,batch_size*300,layer0_out,layer0_a_out);
		cpu_gemm(batch_size,300,300,layer0_a_out,layer1,layer1_out,bias1);
		//cout<<"layer1"<<endl;
		//for (int i=0;i<10;i++){
		//	cout<<layer1_out[i]<<endl;
		//}
        	cpu_activation(0,batch_size*300,layer1_out,layer1_a_out);
		cpu_gemm(batch_size,300,300,layer1_a_out,layer2,layer2_out,bias2);
		//cout<<"layer2"<<endl;
		//for (int i=0;i<10;i++){
		//	cout<<layer2_out[i]<<endl;
		//}
        	cpu_activation(0,batch_size*300,layer2_out,layer2_a_out);

		cpu_gemm(batch_size,1,300,layer2_a_out,concat_prj,concat_out,concat_bias);
		cpu_activation(1,batch_size,concat_out,concat_a_out);
		gettimeofday(&p_stop,NULL);
		gettimeofday(&io_stop,NULL);

		float ycl_msec = (io_start.tv_sec-ycl_start.tv_sec)*1000.0+(io_start.tv_usec-ycl_start.tv_usec)/1000.0;
                float io_msec = (p_start.tv_sec-io_start.tv_sec)*1000.0+(p_start.tv_usec-io_start.tv_usec)/1000.0;
                io_msec = (io_stop.tv_sec-p_stop.tv_sec)*1000.0+(io_stop.tv_usec-p_stop.tv_usec)/1000.0;
                float p_msec = (p_stop.tv_sec-p_start.tv_sec)*1000.0+(p_stop.tv_usec-p_start.tv_usec)/1000.0;
                cout<<"ycl time: "<<ycl_msec<<" io time: "<<io_msec<<" polaris time: "<<p_msec<<endl;	

		for (int i=0;i<batch_size;i++){
			fres<<concat_a_out[i]<<endl;
		}	
	}
	cout<<"end"<<endl;
	return 0;
}
