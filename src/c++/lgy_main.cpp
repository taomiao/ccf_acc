#include<cstdlib>
#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<sys/time.h>
#include"cpu_lib.h"

#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>

using namespace std;

int NUM_THREADS = 10;
int * test_data;
int * test_feat;
long long batch_size = 200000;
float * embeddings = NULL;
float * embedding_data = NULL;
int * batch_test_data = NULL;
float * trans_mul_data = NULL;

float * load_param_by_name(char * filename,int size){
        float * data = (float*)malloc(size*sizeof(float));
	string file_url("./params2/");
	file_url += filename;
        ifstream param_file(file_url.c_str());
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
	return data;
}

int load_test_data(string test_name){
	ifstream infd(test_name.c_str());
	string line;
	int cnt=0;
	int * p = test_data;
	while(getline(infd,line)){
		istringstream sin(line);
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
        return 0;
}

void* run(void* num){
    long long st = *((long long*)num);
    st = st*(batch_size/NUM_THREADS);
    long long ed = st+batch_size/NUM_THREADS;
	float * embd = NULL;
	embd = embeddings;
	float * result = embedding_data;
	long long res_idx=st*30*40*40;
	for(long long k=st;k<ed;k++){
	float *input = result+res_idx;
	
		for(long long i=0;i<40;i++){
			int idx = batch_test_data[k*39+i];
//cout<<"idx"<<endl;
			if(i==39)
				idx=12460;
			float * start_pos = embd+1200*idx;
			for(int j=0;j<1200;j++){
				result[res_idx]=*start_pos;
				res_idx++;
				start_pos += 1;	
			}
		}
		long long out_idx = 20*39*30*st;
		for (long long m = 0;m<40;m++){
			for(long long n=0;n<40;n++){
				if(m>=n){
					continue;
				}
				for( long long ki=0;ki<30;ki++){
					float tmp1 =(*(input+(n*40*30+m*30+ki)));
					float tmp2 =(*(input+(m*40*30+n*30+ki)));
					trans_mul_data[out_idx] = tmp1*tmp2;
					out_idx ++;
//		cout<<"out_idx"<<out_idx;
				}
			}
		}
	}
}

int main(){
	test_data = (int*)malloc(1000000*39*sizeof(int));

	long long params[NUM_THREADS];
	embeddings = load_param_by_name("feature_embeddings.csv",12461*1200);

	float * layer0 = load_param_by_name("layer_0_t.csv",23400*300);
	float * bias0 = load_param_by_name("bias_0.csv",300);
	float * layer1 = load_param_by_name("layer_1_t.csv",300*300);
	float * bias1 = load_param_by_name("bias_1.csv",300);
	float * layer2 = load_param_by_name("layer_2_t.csv",300*300);
	float * bias2 = load_param_by_name("bias_2.csv",300);
	float * concat_prj = load_param_by_name("concat_projection.csv",300);
	float * concat_bias = load_param_by_name("concat_bias.csv",1);

		
	embedding_data = (float*)malloc(batch_size*40*1200*sizeof(float));
	trans_mul_data = (float *)malloc(batch_size*20*39*30*sizeof(float));
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
	ofstream fres("cpu_result.txt");
	ofstream time_record("cpu_time_record.csv");
	clock_t total_time = 0;
	for( int i=0;i<1000000/batch_size;i++){
		batch_test_data = test_data + (batch_size*39*i);
		cout<<"pre-handling data ..."<<endl;
		
		timeval ycl_start,io_start,p_start,p_stop,io_stop;
		
		gettimeofday(&ycl_start,NULL);
		
        pthread_t tids[NUM_THREADS];
        for(int i = 0; i < NUM_THREADS; i++){
            	params[i] = i;
		pthread_create(&tids[i], NULL, run, (void *)&(params[i]));
        }
		for(int i = 0; i < NUM_THREADS; i++){
            pthread_join (tids[i], NULL );
		}

		cpu_gemm(batch_size,300,23400,trans_mul_data,layer0,layer0_out,bias0);
		gettimeofday(&io_start,NULL);
		gettimeofday(&p_start,NULL);
		cpu_activation(0,batch_size*300,layer0_out,layer0_a_out);
		cpu_gemm(batch_size,300,300,layer0_a_out,layer1,layer1_out,bias1);
        cpu_activation(0,batch_size*300,layer1_out,layer1_a_out);
		cpu_gemm(batch_size,300,300,layer1_a_out,layer2,layer2_out,bias2);
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
