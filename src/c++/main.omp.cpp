#include<cstdlib>
#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<omp.h>
#include<sys/time.h>
#include"polaris/include/polaris.h"
//#include"cpu_lib.h"

using namespace std;

int * test_data;
int * test_feat;
int batch_size = 10000;
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

void dumpfloat(char * filename,float * arr,int size){


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
		//cout<<cnt<<endl;
	}
	//cout<<cnt<<endl;
	return 0;
}

void load_whole_test(){
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
	//float *	embd2 = embeddings2;
	float * result = embedding_data;
	//result = (float*)malloc(1024*1800*60*sizeof(float));
	int res_idx=0;
#pragma opm parallel for
	for(int k=0;k<batch_size;k++){
		for(int i=0;i<40;i++){
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
	//return result;
}

void trans_and_mul(float * input,float * output){
	//input 1024 * 60 * 60 * 30
	int idx = 0;
#pragma omp parallel for
	for (int i=0;i<batch_size;i++){
		for ( int m = 0;m<40;m++){
			for(int n=0;n<40;n++){
				if(m>=n){
					continue;
				}
				for( int k=0;k<30;k++){
					float tmp1 =(*(input+(i*40*40*30+n*40*30+m*30+k)));
					float tmp2 =(*(input+(i*40*40*30+m*40*30+n*30+k)));
					output[idx] = tmp1*tmp2;
	//				cout<<idx<<endl;
					idx ++;
				}
			}
		}
	}
}

clock_t total_time = 0;

int main(){
	//cout<<"hello world"<<endl;
	ofstream time_record("fpga_time_record.csv");	
	//test_data = (int*)malloc(batch_size*40*sizeof(int));
	cout<<"loading original data..."<<endl;
	test_data = (int*)malloc(1000000*39*sizeof(int));
	load_whole_test();
	cout<<"finish loading original data."<<endl;
	//test_feat = (int*)malloc(batch_size*20*sizeof(int));
		
	cout<<"loading params ..."<<endl;
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

	float * layer0 = load_param_by_name("layer_0_t.csv",23400*300);
	float * bias0 = load_param_by_name("bias_0.csv",300);
	float * layer1 = load_param_by_name("layer_1_t.csv",300*300);
	float * bias1 = load_param_by_name("bias_1.csv",300);
	float * layer2 = load_param_by_name("layer_2_t.csv",300*300);
	float * bias2 = load_param_by_name("bias_2.csv",300);
	float * concat_prj = load_param_by_name("concat_projection.csv",300);
	float * concat_bias = load_param_by_name("concat_bias.csv",1);
	
	cout<<"finish loading params."<<endl;
	cout<<"init fpga..."<<endl;
	float * embedding_data = (float *)malloc(batch_size*40*40*30*sizeof(float));
	float * trans_mul_data = (float *)malloc(batch_size*23400*sizeof(float));
	float * result = (float *)malloc(batch_size*sizeof(float));
	
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

	polaris_malloc(ctx,batch_size*23400*sizeof(float),(void**)&fpga_test_data);
	polaris_malloc(ctx,23400*300*sizeof(float),(void**)&fpga_layer0);
	polaris_malloc(ctx,300*sizeof(float),(void**)&fpga_bias0);
	polaris_malloc(ctx,batch_size*300*sizeof(float),(void**)&fpga_layer0_out);
	polaris_malloc(ctx,batch_size*300*sizeof(float),(void**)&fpga_layer0_a_out);
	polaris_malloc(ctx,300*300*sizeof(float),(void**)&fpga_layer1);
	polaris_malloc(ctx,300*sizeof(float),(void**)&fpga_bias1);
	polaris_malloc(ctx,batch_size*300*sizeof(float),(void**)&fpga_layer1_out);
	polaris_malloc(ctx,batch_size*300*sizeof(float),(void**)&fpga_layer1_a_out);
	polaris_malloc(ctx,300*300*sizeof(float),(void**)&fpga_layer2);
	polaris_malloc(ctx,300*sizeof(float),(void**)&fpga_bias2);
	polaris_malloc(ctx,batch_size*300*sizeof(float),(void**)&fpga_layer2_out);
	polaris_malloc(ctx,batch_size*300*sizeof(float),(void**)&fpga_layer2_a_out);
	polaris_malloc(ctx,300*sizeof(float),(void**)&fpga_concat_prj);
	polaris_malloc(ctx,1*sizeof(float),(void**)&fpga_concat_bias);
	polaris_malloc(ctx,batch_size*sizeof(float),(void**)&fpga_result);
	polaris_malloc(ctx,batch_size*sizeof(float),(void**)&fpga_a_result);
	polaris_malloc(ctx,batch_size*sizeof(float),(void**)&fpga_zero);

	//polaris_memcpy(ctx,POLARIS_HOST_TO_DEVICE,fpga_test_data,test_data,1024*53100*sizeof(float));
	polaris_memcpy(ctx,POLARIS_HOST_TO_DEVICE,fpga_layer0,layer0,23400*300*sizeof(float));
	polaris_memcpy(ctx,POLARIS_HOST_TO_DEVICE,fpga_bias0,bias0,300*sizeof(float));
	polaris_memcpy(ctx,POLARIS_HOST_TO_DEVICE,fpga_layer1,layer1,300*300*sizeof(float));
	polaris_memcpy(ctx,POLARIS_HOST_TO_DEVICE,fpga_bias1,bias1,300*sizeof(float));
	polaris_memcpy(ctx,POLARIS_HOST_TO_DEVICE,fpga_layer2,layer2,300*300*sizeof(float));
	polaris_memcpy(ctx,POLARIS_HOST_TO_DEVICE,fpga_bias2,bias2,300*sizeof(float));		
	polaris_memcpy(ctx,POLARIS_HOST_TO_DEVICE,fpga_concat_prj,concat_prj,300*sizeof(float));
	polaris_memcpy(ctx,POLARIS_HOST_TO_DEVICE,fpga_concat_bias,concat_bias,1*sizeof(float));
	//polaris_memset(ctx,
	//
	cout<<"finish init fpga."<<endl;

	ofstream fres("fpga_result.txt");

	//int * batch_test_data = (int*)malloc(batch_size*40*sizeof(int));
	//计算过程
	cout<<"batch size: "<<batch_size<<endl;
	for ( int i=0;i<1000000/batch_size;i++ ){
		//load_test_data(test_file_name);
		//cout<<"------------------ok"<<endl;
		//char num[5];
		//sprintf(num,"%05d",i);
		//string s(num);
		//cout<<s<<endl;
		//load_test_data("./TEST_B_csv/TEST_"+s);
		//load_test_feat("./TEST_feat/TEST_"+s);
		//cout<<"------------------ok"<<endl;

		cout<<"prehandling data for batch "<<i<<endl;
		int * batch_test_data = test_data+(batch_size*39*i);
		
		timeval ycl_start,io_start,p_start,p_stop,io_stop;

		gettimeofday(&ycl_start,NULL);
		embedding_lookup(batch_test_data,embedding_data); //1024*60*1800
		trans_and_mul(embedding_data,trans_mul_data);
		gettimeofday(&io_start,NULL);
	
		//cout<<"finish prehandling data."<<endl;
		//cout<<"ok! finish loading test file"<<endl;
		//trans_mul_data = load_param_by_name("split_mid_feature_0.csv",1024*53100);
		//cout<<"copy data to fpga>>>>"<<endl;

		polaris_memcpy(ctx,POLARIS_HOST_TO_DEVICE,fpga_test_data,trans_mul_data,batch_size*23400*sizeof(float));
		
		gettimeofday(&p_start,NULL);
		//cout<<"finish copy data with time "<<(cp_finish-cp_start)/1000000.0<<" s"<<endl;
		//time_record<<cp_finish-cp_start<< ",";
		
		//cout<<"computing one batch..."<<endl;
		//clock_t start = clock();
		polaris_gemm(ctx,batch_size,300,23400,fpga_test_data,fpga_layer0,fpga_layer0_out,fpga_bias0);
		polaris_activation(ctx,POLARIS_RELU,batch_size*300,1,fpga_layer0_out,0,fpga_layer0_a_out);
		
		//cout<<"tmp_l0 ....................."<<endl;
		//float * tmp_l0 = NULL;
		//tmp_l0 = (float*)malloc(1024*300*sizeof(float));
		//polaris_memcpy(ctx,POLARIS_DEVICE_TO_HOST,tmp_l0,fpga_layer0_out,1024*300*sizeof(float));
		//for(int j=0;j<10;j++){
		//		cout<<tmp_l0[j]<<" , ";
		//}

		polaris_gemm(ctx,batch_size,300,300,fpga_layer0_a_out,fpga_layer1,fpga_layer1_out,fpga_bias1);
		
		//cout<<"tmp_l1 ......................"<<endl;
		//polaris_memcpy(ctx,POLARIS_DEVICE_TO_HOST,tmp_l0,fpga_layer1_out,1024*300*sizeof(float));
		//for(int j=0;j<10;j++){
		//	cout<<tmp_l0[j]<<",";
		//}
			
		polaris_activation(ctx,POLARIS_RELU,batch_size*300,1,fpga_layer1_out,0,fpga_layer1_a_out);
		polaris_gemm(ctx,batch_size,300,300,fpga_layer1_a_out,fpga_layer2,fpga_layer2_out,fpga_bias2);
		polaris_activation(ctx,POLARIS_RELU,batch_size*300,1,fpga_layer2_out,0,fpga_layer2_a_out);
		polaris_gemm(ctx,batch_size,1,300,fpga_layer2_a_out,fpga_concat_prj,fpga_result,fpga_concat_bias);
	

		//cout<<"tmp_prj ......................"<<endl;
		//polaris_memcpy(ctx,POLARIS_DEVICE_TO_HOST,tmp_l0,fpga_result,1024*sizeof(float));
		//for(int j=0;j<10;j++){
		//	cout<<tmp_l0[j]<<","<<endl;
		//}
	
		polaris_activation(ctx,POLARIS_SIGMOID,batch_size,1,fpga_result,0,fpga_a_result);
		
		//clock_t stop = clock();
		gettimeofday(&p_stop,NULL);
		//total_time += (stop-start);
		//cout<<"finish. time costs: "<<msec<<" ms"<<endl;
		//time_record<<msec<<",";

		//cout<<"copy result back to main memory..."<<endl;
		
		//clock_t cpb_start = clock();	
		polaris_memcpy(ctx,POLARIS_DEVICE_TO_HOST,result,fpga_a_result,batch_size*sizeof(float));
		//clock_t cpb_stop = clock();
		gettimeofday(&io_stop,NULL);

		float ycl_msec = (io_start.tv_sec-ycl_start.tv_sec)*1000.0+(io_start.tv_usec-ycl_start.tv_usec)/1000.0;
                float io_msec = (p_start.tv_sec-io_start.tv_sec)*1000.0+(p_start.tv_usec-io_start.tv_usec)/1000.0;
                io_msec = (io_stop.tv_sec-p_stop.tv_sec)*1000.0+(io_stop.tv_usec-p_stop.tv_usec)/1000.0;
                float p_msec = (p_stop.tv_sec-p_start.tv_sec)*1000.0+(p_stop.tv_usec-p_start.tv_usec)/1000.0;
                cout<<"ycl time: "<<ycl_msec<<" io time: "<<io_msec<<" polaris time: "<<p_msec<<endl;	
		//cout<<"finish copy back. time cost: "<<(cpb_stop-cpb_start)/1000000.0<<" s"<<endl;
		//time_record<<cpb_stop-cpb_start<<endl;
	//	float * res = (float*)malloc(1024*sizeof(float));
	//	cpu_activation(1,1024,result,res);
		for(int j=0;j<batch_size;j++){
	//		float tmp = 0;
	//		tmp = res[j];
			fres.precision(6);
			fres<<result[j]<<endl;
		}
	}	

	polaris_destroy_context(ctx);
//	cout<<"end"<<endl;
	return 0;
}
