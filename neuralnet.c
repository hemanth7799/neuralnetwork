#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>

float X2[20],z[20],Wji[20][20],Wkj[20][20],D_Wkj_ofNew[20][20],D_Wji_ofNew[20][20],D_Wkj[20][20],D_Wji[20][20];
float norm_calculation(int nrow, int ncol,float A[][ncol]);
float RandomNumber(float Min, float Max);
void backwardpropogtion(float t[], float X1[], float eta);
float forwardpropogtion(float X1[],int n);
float sigmoidfunction(float x);

int main(){
	int accuracy;
     srand(time(NULL));
    for(int i=0;i<17;i++){
    	for(int j=0;j<5;j++)
    		Wji[i][j]=RandomNumber(0.01,0.09); 	
    }
    float norm_ofWji;
    norm_ofWji=norm_calculation(17,5,D_Wji_ofNew);
    for(int j=0;j<6;j++){
        for(int k=0;k<10;k++)
            Wkj[j][k]=RandomNumber(0.01,0.09);  
    }
    printf("training neural network with epochs\n");
    float norm_ofWkj=norm_calculation(6,10,D_Wkj_ofNew);
   //float epsilon = 0.01;// assigning epsilon
    FILE *f = fopen("train1.txt","r");
		float A[3000][50];
		while(getc(f)!=EOF){
			for(int i=0;i<2216;i++){
				for(int j=0;j<17;j++)
					fscanf(f,"%f",&A[i][j]);		
			}}
			float avg_norm=abs(norm_ofWji-norm_ofWkj);
    int epho=0;
	while(epho<10000){
			for(int i=0;i< 2216;i++){
				float X[20];
				for(int j=1;j<17;j++)
					X[j-1]=A[i][j];	
				//creation of  target output
				int class_label=(int)A[i][0];
				float T[15];
				int j=0;
				while(j<10){T[j]=0.0;j++;}
				T[class_label-1]=1.0;
				forwardpropogtion(X,5);
				backwardpropogtion(T,X,0.001);
				for(int l=0;l<17;l++){
                  for(int j=0;j<5;j++)
                      Wji[l][j]=Wji[l][j]+D_Wji[l][j];   
                 }
				for(int j=0;j<6;j++){
                   for(int k=0;k<10;k++)
                      Wkj[j][k]=Wkj[j][k]+D_Wkj[j][k]; 
                  }      
			}	      	
		norm_ofWkj = norm_calculation(6,10,D_Wkj_ofNew);
		norm_ofWji = norm_calculation(17,5,D_Wji_ofNew);
		avg_norm = abs(norm_ofWji-norm_ofWkj); 
        epho=epho+1; 
    }  
         printf("TRAINING COMPLETE...............\n");
	 FILE *f1= fopen("test.txt", "r");
		float Atest[3000][50];
		while(getc(f1)!= EOF){
			for(int i=0;i<999;i++){
				for(int j=0;j<17;j++){
					fscanf(f1,"%f",&Atest[i][j]);
				}}}
		int label[999];
		for(int i=0;i<999;i++){
		   float Xtest[20];
				for(int j=1;j<17;j++)
					Xtest[j-1]=Atest[i][j];
		   forwardpropogtion(Xtest,5);
		   float max=INT_MIN *1.0;
		   int index;
		   for(int p=0;p<10;p++){
		   		 if(z[p]>max){
		   			max=z[p];
		   			index=p;
		   		}
		   }		   
		   label[i]=index+1;		   
	if(index+1==Atest[i][0]){accuracy++;}
           }   
           accuracy=accuracy/10;
           printf("accuracy is:%d\n",accuracy);   	
	return 0;
}
float sigmoidfunction(float x){
	float y,exp_value;
	exp_value=exp(-x);
	y=1/(1+exp_value);
	return y;
}
float forwardpropogtion(float X1[],int n){
	//sigmoidfunction calculation for hiddenlayer
	float X1_bias[20];
	X1_bias[0]=1.0;
	for(int j=1;j<17;j++)
		X1_bias[j]=X1[j-1];
	for(int j=0;j<5;j++){
	    float sum=0;
		for(int i=0;i<17;i++)
			sum=sum+X1_bias[i]*Wji[i][j];	
		X2[j]=sigmoidfunction(sum);
	}	
	//sigmoidfunction calculation for output layer
	float X2_bias[20];
	X2_bias[0]=1.0;
	for(int k=1;k<5;k++)
		X2_bias[k]=X2[k-1];
	for(int k=0;k<10;k++){
	    float sum1=0;
		for(int j=0;j<6;j++)
		   sum1=sum1+X2_bias[j]*Wkj[j][k];  
		z[k]=sigmoidfunction(sum1);
	}
}
void backwardpropogtion(float t[], float X1[], float eta){
        //process between hidden and output layer
	    float delta_output [20]; // error for the output layer
	    //updating the values of error in delta_output
	    for(int k=0; k<10; k++)
	    	delta_output[k] =  ( t[k]-z[k] ) * ( z[k]*(1.0-z[k]) );
		float X2_bias[20];
	    X2_bias[0]=1.0;
	    for(int k=1;k<5;k++)
		   X2_bias[k]=X2[k-1];
		for(int k=0; k<10; k++){
		   for(int j=0; j<6; j++)
		  	D_Wkj[j][k] = eta * X2_bias[j] * delta_output[k];	
		}
        //process between input and hidden layer
		float delta_hidden[10]; // error for the hidden layer
	    //updating the values of error in delta_hidden
		for(int j=1; j<6; j++){
			float sum = 0;
			for(int r=0; r<10; r++)
				sum = sum+( delta_output[r] * Wkj[j][r] * ( X2[j-1]*(1.0-X2[j-1]) ) );	
			delta_hidden[j-1] = sum;
		}
		//update of Delta weight between input and hidden layer  /_\Wji
        float X1_bias[20];
	    X1_bias[0]=1.0;
	    int h_index;

	    for(int j=1;j<17;j++)
		    X1_bias[j]=X1[j-1];
		for(int j=0;j<5;j++){
			for(int i=0;i<17;i++)
				D_Wji[i][j]=eta*X1_bias[i]*delta_hidden[j];	
		}}
float RandomNumber(float Min,float Max){
    //return (((float)rand() / (float)RAND_MAX) * (Max - Min)) + Min;
	return ((float)(rand()%2-2)/1);
}
float norm_calculation(int nrow, int ncol,float A[][ncol]){
	int i=0,j;
	float norm,sum=0;
	while(i<nrow){
		for(j=0;j<ncol;j++)
			sum=sum+(A[i][j]*A[i][j]);
		i=i+1;
	}
    norm=sqrt((float)sum);
    return (norm);    
}
