/*
The output filename generated from dataset_generator.cpp has been renamed to "expected.raw"
Currently, the dataset generates a vector of 35 single-precision floating point nos. The no. of elements 
can be edited through dataset_generator.cpp

Run this program by first compiling dataset_generator.cpp
Then do "./a.out input0.raw input1.raw output.raw"
After that, open the "output.raw" file to check for computational correctness

*/

#include<cstdio>
#include<iostream>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>

int main(int argc, char** argv)
{
	/* ip1: input0.raw
	   ip2: input1.raw
	   op: output.raw
	*/
	
	FILE* ip1=fopen(argv[1],"r");
	FILE* ip2=fopen(argv[2],"r");
	FILE* op=fopen(argv[3],"w");
	
	/*Get the number of elements from both the vectors (count1==count2)*/
	int count1,count2;
	fscanf(ip1,"%d",&count1);
	fscanf(ip2,"%d",&count2);
	
	/*Declare arr1 and arr2 host vectors*/
	thrust::host_vector<float> arr1;
	thrust::host_vector<float> arr2;
	
	/*Read elements from ip1 and ip2. Append them to arr1 and arr2*/
	float temp1,temp2;
	for(int i=0;i<count1;i++)
	{
		fscanf(ip1,"%f",&temp1);
		fscanf(ip2,"%f",&temp2);
		
		arr1.push_back(temp1);
		arr2.push_back(temp2);
	}
	
	/*Copy the host vectors to the respective device vectors. Declare a device vector sum_d of size=count1*/
	thrust::device_vector<float> arr1_d=arr1;
	thrust::device_vector<float> arr2_d=arr2;
	thrust::device_vector<float> sum_d(count1);
	
	/*Perform vector addition by invoking thrust::transform*/
	thrust::transform(arr1_d.begin(),arr1_d.end(),arr2_d.begin(),sum_d.begin(),thrust::plus<float>());
	/*Copy the device vector sum_d to sum*/
	thrust::host_vector<float> sum=sum_d;
	
	/*Print out all the elements of arr1,arr2 and sum in the "output.raw" file*/
	for(int i=0;i<count1;i++)
	{
		fprintf(op,"\n%5.2f %5.2f %5.2f",arr1[i],arr2[i],sum[i]);
	}
	
	fclose(ip1);
	fclose(ip2);
	fclose(op);
}