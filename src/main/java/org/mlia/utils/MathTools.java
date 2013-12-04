package org.mlia.utils;

import java.util.Arrays;

public class MathTools {

	public static double dotProduct(double[] v1,double[] v2){
		if(v1.length!=v2.length){
			throw new RuntimeException("vector's size don't match");
		}
		
		double result=0;
		for(int i=0;i<v1.length;i++){
			result+=(v1[i]*v2[i]);
		}
		
		return result;
	}
	
	public static double[] expandVec(double[] vec1,double[] vec2){
		double[] v=new double[vec1.length+vec2.length];
		int i=0;
		for(;i<vec1.length;i++){
			v[i]=vec1[i];
		}
		for(int j=0;j<vec2.length;j++,i++){
			v[i]=vec2[j];
		}
		return v;
	}
	
	public static double[] expandVec(double[] vec1,double number){
		double[] v=new double[vec1.length+1];
		int i=0;
		for(;i<vec1.length;i++){
			v[i]=vec1[i];
		}
		v[i]=number;
		return v;
	}

	public static double[][] transposeMatrix(double[][] matrix){
		double[][] result=new double[matrix[0].length][matrix.length];
		for(int i=0;i<matrix.length;i++){
			for(int j=0;j<matrix[i].length;j++){
				result[j][i]=matrix[i][j];
			}
		}
		
		return result;
	}
	
	public static double sigmoid(double x){
		return 1.0/(1+Math.exp(-x));
	}
	
	public static double[] vecSubtraction(double[] v1, double[] v2){
		if(v1.length!=v2.length){
			throw new RuntimeException("vector's size don't match");
		}
		double[] v=new double[v1.length];
		for(int i=0;i<v.length;i++){
			v[i]=v1[i]-v2[i];
		}
		return v;
	}
	
	public static double[] vecAddition(double[] v1,double[] v2){
		if(v1.length!=v2.length){
			throw new RuntimeException("vector's size don't match");
		}
		double[] v=new double[v1.length];
		for(int i=0;i<v.length;i++){
			v[i]=v1[i]+v2[i];
		}
		return v;		
	}
	
	public static void vecAdditionOnthefly(double[] v1,double[] v2){
		if(v1.length!=v2.length){
			throw new RuntimeException("vector's size don't match");
		}
 
		for(int i=0;i<v1.length;i++){
			v1[i]+=v2[i];
		}		
	}
	
	public static void vecSubtractionOnthefly(double[] v1,double[] v2){
		if(v1.length!=v2.length){
			throw new RuntimeException("vector's size don't match");
		}		
		
		for(int i=0;i<v1.length;i++){
			v1[i]-=v2[i];
		}
	}
	
	public static double[] copyVec(double[] vec){

		return Arrays.copyOf(vec, vec.length);
	}
	
	public static double[] vecMultiplyNumber(double[] vec,double a){
		double[] result=new double[vec.length];
		for(int i=0;i<vec.length;i++){
			result[i]=vec[i]*a;
		}
		return result;
	}
	
	public static void vecMultiplyNumberOnthefly(double[] vec,double a){
		for(int i=0;i<vec.length;i++){
			vec[i]*=a;
		}
	}
	
	public static double[][] matrixMultiply(double[][] matrix1, double[][] matrix2){
		if(matrix1[0].length!=matrix2.length){
			throw new RuntimeException("matrix1's col <. matrix2's row");
		}
		double[][] result=new double[matrix1.length][matrix2[0].length];
		for(int i=0;i<result.length;i++){	
			for(int j=0;j<result[0].length;j++){
				double s=0;
				for(int k=0;k<matrix1[0].length;k++){
					s+=matrix1[i][k]*matrix2[k][j];
				}
				result[i][j]=s;
			}
		}
		
		return result;
	}
	
	public static double[] maxtrixMultiply(double[][] matrix,double[] vec){
		if(matrix[0].length!=vec.length){
			throw new RuntimeException("matrix's col <. vec's row");
		}
		
		double[] result=new double[matrix.length];
		for(int i=0;i<result.length;i++){
			double sum=0;
			for(int j=0;j<vec.length;j++){
				sum+=matrix[i][j]*vec[j];
			}
			result[i]=sum;
		}
		
		return result;
	}	
}
