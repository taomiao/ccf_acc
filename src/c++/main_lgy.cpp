#include<cstdlib>
#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include"polaris/include/polaris.h"

using namespace std;

int * test_data;
int * test_feat;
float * embeddings = NULL;
float * embeddings2 = NULL;

float * load_param_by_name(char * filename,int size){
	cout<<"loading "<<filename<<endl;
        float * data = (float*)malloc(size*sizeof(float));
	string file_url("./params/");
	file_url += filename;
        ifstream param_file(file_url.c_str());
        cout<<"ok"<<endl;
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
        cout<<cnt<<endl;
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
	cout<<cnt<<endl;
	return 0;
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
        cout<<cnt<<endl;
        return 0;
}

float * embedding_lookup(int * d_input,int * f_input){
	float * embd = NULL;
	embd = embeddings;
	float *	embd2 = embeddings2;
	float * result = NULL;
	result = (float*)malloc(1024*1800*60*sizeof(float));
	int res_idx=0;
	for(int k=0;k<1024;k++){
		for(int i=0;i<40;i++){
			int idx = d_input[k*40+i];
			float * start_pos = embd+1800*idx;
			for(int j=0;j<1800;j++){
				result[res_idx]=*start_pos;
				res_idx++;
				start_pos += 1;	
			}
		}
		for(int i=0;i<20;i++){
			int idx = f_input[k*20+i];
			float * start_pos = embd2+1800*idx;
                        for(int j=0;j<1800;j++){
                                result[res_idx]=*start_pos;
                                res_idx++;
                                start_pos += 1;
                        }
		}
	}
	return result;
}

void trans_and_mul(float * input,float * output){
	//input 1024 * 60 * 60 * 30
	int idx = 0;
	for (int i=0;i<1024;i++){
		for ( int m = 0;m<60;m++){
			for(int n=0;n<60;n++){
				if(m>=n){
					continue;
				}
				for( int k=0;k<30;k++){
					float tmp1 =(*(input+(i*60*60*30+n*60*30+m*30+k)));
					float tmp2 =(*(input+(i*60*60*30+m*60*30+n*30+k)));
					output[idx] = tmp1*tmp2;
	//				cout<<idx<<endl;
					idx ++;
				}
			}
		}
	}
}


int main(){
	cout<<"hello world"<<endl;
	test_data = (int*)malloc(1024*40*sizeof(int));
	test_feat = (int*)malloc(1024*20*sizeof(int));
		
	cout<<"------------------ok"<<endl;
	embeddings = load_param_by_name("feature_embeddings.csv",12461*1800);
	embeddings2 = load_param_by_name("feature_embeddings2.csv",10000*1800);
	
	cout<<"------------------ok"<<endl;
	load_test_data("./TEST_csv/TEST_00000");
	load_test_feat("./TEST_feat/TEST_00000");
	cout<<"------------------ok"<<endl;
	float * test_data_embd = embedding_lookup(test_data,test_feat); //1024*60*1800
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

	float * trans_mul_data = (float *)malloc(1024*30*59*30*sizeof(float));
	trans_and_mul(test_data_embd,trans_mul_data);
	
	cout<<"mid_file start -----------------"<<endl;
	for(int i=0;i<100;i++){
		cout<<trans_mul_data[1024*30*59*30-1-i]<<endl;
	}

	float * layer0 = load_param_by_name("layer_0.csv",53100*300);
	float * bias0 = load_param_by_name("bias_0.csv",300);
	float * layer1 = load_param_by_name("layer_1.csv",300*300);
	float * bias1 = load_param_by_name("bias_1.csv",300);
	float * layer2 = load_param_by_name("layer_2.csv",300*300);
	float * bias2 = load_param_by_name("bias_2.csv",300);
	float * concat_prj = load_param_by_name("concat_projection.csv",300);
	float * concat_bias = load_param_by_name("concat_bias.csv",1);
	float * result = (float *)malloc(1024*sizeof(float));
	
	float * fpga_test_data = NULL; //每次1000条数据
	float * fpga_layer0 = NULL;
	float * fpga_bias0 = NULL;

	float * fpga_layer0_out = NULL;
	float * fpga_layer0_a_out = NULL;	

	float * fpga_layer1 = NULL;
	float * fpga_bias1 = NULL;

	float * fpga_layer1_out = NULL;
	float * fpga_layer1_a_out = NULL;

	float * fpga_layer2 = NULL;
	float * fpga_bias2 = NULL;

	float * fpga_layer2_out = NULL;
	float * fpga_layer2_a_out = NULL;

	float * fpga_concat_prj = NULL;
	float * fpga_concat_bias = NULL;
	
	float * fpga_result = NULL;
	float * fpga_a_result = NULL;
	
	float * fpga_zero = NULL;

	PolarisContext * ctx = polaris_create_context(0);

	polaris_malloc(ctx,1024*53100*sizeof(float),(void**)&fpga_test_data);
	polaris_malloc(ctx,53100*300*sizeof(float),(void**)&fpga_layer0);
	polaris_malloc(ctx,300*sizeof(float),(void**)&fpga_bias0);
	polaris_malloc(ctx,1024*300*sizeof(float),(void**)&fpga_layer0_out);
	polaris_malloc(ctx,1024*300*sizeof(float),(void**)&fpga_layer0_a_out);
	polaris_malloc(ctx,300*300*sizeof(float),(void**)&fpga_layer1);
	polaris_malloc(ctx,300*sizeof(float),(void**)&fpga_bias1);
	polaris_malloc(ctx,1024*300*sizeof(float),(void**)&fpga_layer1_out);
	polaris_malloc(ctx,1024*300*sizeof(float),(void**)&fpga_layer1_a_out);
	polaris_malloc(ctx,300*300*sizeof(float),(void**)&fpga_layer2);
	polaris_malloc(ctx,300*sizeof(float),(void**)&fpga_bias2);
	polaris_malloc(ctx,1024*300*sizeof(float),(void**)&fpga_layer2_out);
	polaris_malloc(ctx,1024*300*sizeof(float),(void**)&fpga_layer2_a_out);
	polaris_malloc(ctx,300*sizeof(float),(void**)&fpga_concat_prj);
	polaris_malloc(ctx,1*sizeof(float),(void**)&fpga_concat_bias);
	polaris_malloc(ctx,1024*sizeof(float),(void**)&fpga_result);
	polaris_malloc(ctx,1024*sizeof(float),(void**)&fpga_a_result);
	polaris_malloc(ctx,1024*sizeof(float),(void**)&fpga_zero);

	//polaris_memcpy(ctx,POLARIS_HOST_TO_DEVICE,fpga_test_data,test_data,1024*53100*sizeof(float));
	polaris_memcpy(ctx,POLARIS_HOST_TO_DEVICE,fpga_layer0,layer0,53100*300*sizeof(float));
	polaris_memcpy(ctx,POLARIS_HOST_TO_DEVICE,fpga_bias0,bias0,300*sizeof(float));
	polaris_memcpy(ctx,POLARIS_HOST_TO_DEVICE,fpga_layer1,layer1,300*300*sizeof(float));
	polaris_memcpy(ctx,POLARIS_HOST_TO_DEVICE,fpga_bias1,bias1,300*sizeof(float));
	polaris_memcpy(ctx,POLARIS_HOST_TO_DEVICE,fpga_layer2,layer2,300*300*sizeof(float));
	polaris_memcpy(ctx,POLARIS_HOST_TO_DEVICE,fpga_bias2,bias2,300*sizeof(float));		
	polaris_memcpy(ctx,POLARIS_HOST_TO_DEVICE,fpga_concat_prj,concat_prj,300*sizeof(float));
	polaris_memcpy(ctx,POLARIS_HOST_TO_DEVICE,fpga_concat_bias,concat_bias,1*sizeof(float));
	//polaris_memset(ctx,
	
	//计算过程
	for ( int i=0;i<1;i++ ){
		//load test_data i
		//string test_file_name("./split_mid_feature_0.csv");
		//test_file_name += (i+'0');
		//test_file_name += ".csv";
		//cout<<"loading test file"<<test_file_name<<endl;
		//load_test_data(test_file_name);
		cout<<"ok! finish loading test file"<<endl;
		polaris_memcpy(ctx,POLARIS_HOST_TO_DEVICE,fpga_test_data,trans_mul_data,1024*53100*sizeof(float));
		polaris_gemm(ctx,1024,300,53100,fpga_test_data,fpga_layer0,fpga_layer0_out,fpga_bias0);
		polaris_activation(ctx,POLARIS_RELU,1024*300,1,fpga_layer0_out,0,fpga_layer0_a_out);
		polaris_gemm(ctx,1024,300,300,fpga_layer0_a_out,fpga_layer1,fpga_layer1_out,fpga_bias1);	
		polaris_activation(ctx,POLARIS_RELU,1024*300,1,fpga_layer1_out,0,fpga_layer1_a_out);
		polaris_gemm(ctx,1024,300,300,fpga_layer1_a_out,fpga_layer2,fpga_layer2_out,fpga_bias2);
		polaris_activation(ctx,POLARIS_RELU,1024*300,1,fpga_layer2_out,0,fpga_layer2_a_out);
		polaris_gemm(ctx,1024,1,300,fpga_layer2_a_out,fpga_concat_prj,fpga_result,fpga_concat_bias);
		polaris_activation(ctx,POLARIS_SIGMOID,1024,1,fpga_result,0,fpga_a_result);
		
		polaris_memcpy(ctx,POLARIS_DEVICE_TO_HOST,result,fpga_a_result,1024*sizeof(float));
		for(int j=0;j<1024;j++){
			float tmp = 0;
			tmp = result[j];
			cout.precision(6);
			cout<<tmp<<endl;
		}
	}	

	polaris_destroy_context(ctx);
	cout<<"end"<<endl;
	return 0;
}
