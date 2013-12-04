package org.mlia.ch06;

import org.mlia.utils.DoubleArrayDataSet;
import org.mlia.utils.MathTools;

public class Perceptron {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		DoubleArrayDataSet ds=Perceptron.createDataSet();
		System.out.println(ds);
		double[] weightsAndBias=Perceptron.perceptronLearning(ds);
		for(double w:weightsAndBias){
			System.out.print(w+"\t");
		}
		System.out.println();
		
		double[][] data=ds.data;
		String[] labels=ds.labels;
		for(int i=0;i<data.length;i++){
			int classifiedResult=Perceptron.classify(weightsAndBias, data[i]);
			System.out.println("realResult: "+labels[i]+"\tclassifiedResult: "+classifiedResult);
		}
		
		DualPerceptronModelParam modelParams=Perceptron.perceptronDualLearning(ds);
		System.out.println(modelParams);
		
		for(int i=0;i<data.length;i++){
			int classifiedResult=Perceptron.classify(modelParams, data[i]);
			System.out.println("realResult: "+labels[i]+"\tclassifiedResult: "+classifiedResult);
		}		
	}
	
	public static int classify(DualPerceptronModelParam param,double[] x){
		double s=0;
		for(int i=0;i<param.matrix.length;i++){
			s+=(MathTools.dotProduct(param.matrix[i], x));
		}
		s+=param.b;
		if(s>=0) return 1;
		else return -1;
	}
	
	public static int classify(double[] weightsAndBias,double[] x){
		double[] xHat=MathTools.expandVec(x, 1);
		double y=MathTools.dotProduct(weightsAndBias, xHat);
		if(y>=0) return 1;
		else return -1;
	}
	
	public static DualPerceptronModelParam perceptronDualLearning(DoubleArrayDataSet ds){
		int n=ds.getRowNumber();
		double[] alpha=new double[n];
		double b=0;
		
		int errorCount=0;
		double eta=1.0;
		double[][] data=ds.data;
		double[] yVec=new double[ds.labels.length];
		for(int i=0;i<yVec.length;i++){
			yVec[i]=Double.valueOf(ds.labels[i]);
		}
		
		boolean hasError=true;
		double[][] gramMatrix=new double[n][];
		for(int i=0;i<n;i++){
			gramMatrix[i]=new double[n];
			for(int j=0;j<n;j++){
				gramMatrix[i][j]=MathTools.dotProduct(data[i], data[j]);
			}
		}
		while(hasError){
			hasError=false;
			for(int i=0;i<data.length;i++){
 
				double y=yVec[i];
				double s=0;
				for(int j=0;j<n;j++){
					s+=(alpha[j]*yVec[j]*gramMatrix[j][i]);
				}
				s+=b;
				if(y*s<=0){
					errorCount++;
					hasError=true;
					alpha[i]+=eta;
					b+=eta*y;
				}
			}
		}
		System.out.println("errorCount: "+errorCount);
		double[][] matrix=new double[n][];
		for(int i=0;i<n;i++){
			matrix[i]=MathTools.copyVec(data[i]);
			MathTools.vecMultiplyNumberOnthefly(matrix[i], alpha[i]*yVec[i]);
		}
		return new DualPerceptronModelParam(matrix, b);
	}
	
	public static double[] perceptronLearning(DoubleArrayDataSet ds){
		int n=ds.getColNumber();
		double[] weights=new double[n];
		double b=0;
		boolean hasError=true;
		double[][] data=ds.data;
		double[] yVec=new double[ds.labels.length];
		for(int i=0;i<yVec.length;i++){
			yVec[i]=Double.valueOf(ds.labels[i]);
		}
		double eta=1.0;
		int errorCount=0;
		while(hasError){
			hasError=false;
			for(int i=0;i<data.length;i++){
				double[] x=data[i];
				double y=yVec[i];
				if(y*(MathTools.dotProduct(weights, x)+b)<=0){
					errorCount++;
					hasError=true;
					double[] delta=MathTools.vecMultiplyNumber(x, eta*y);
					MathTools.vecAdditionOnthefly(weights, delta);
					b=b+eta*y;
				}
				
			}
			
		}
		System.out.println("errorCount: "+errorCount);
		return MathTools.expandVec(weights, b);
	}
	
	public static DoubleArrayDataSet createDataSet(){
		double[][] data=new double[4][];
		String[] labels=new String[4];
		
		data[0]=new double[]{
			3,
			3
		};
		labels[0]="1";
		
		data[1]=new double[]{
			4,
			3
		};
		labels[1]="1";
		
		data[2]=new double[]{
			1,
			1
		};
		labels[2]="-1";

		data[3]=new double[]{
				4,
				1
			};
		labels[3]="-1";
		return new DoubleArrayDataSet(data, labels);
	}

}

class DualPerceptronModelParam{
	public double[][] matrix;
	public double b;
	
	public DualPerceptronModelParam(double[][] matrix,double b){
		this.matrix=matrix;
		this.b=b;
	}
	
	@Override
	public String toString(){
		StringBuilder sb=new StringBuilder();
		sb.append("matrix:\n");
		for(double[] row:matrix){
			for(double d:row){
				sb.append(d).append("\t");
			}
			sb.append("\n");
		}
		sb.append("bias:\t").append(b).append("\n");
		return sb.toString();
	}
}
