package org.mlia.ch05;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.mlia.utils.DoubleArrayDataSet;
import org.mlia.utils.FileTools;
import org.mlia.utils.MathTools;

public class LogisticRegression {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		DoubleArrayDataSet ds=LogisticRegression.loadDataSet();
		System.out.println(ds);
		
		double[] weights=LogisticRegression.gradAscent(ds);
		for(double weight:weights){
			System.out.print(weight+"\t");
		}
		System.out.println();
		weights=LogisticRegression.stocGradAscent0(ds);
		for(double weight:weights){
			System.out.print(weight+"\t");
		}
		System.out.println();	
		weights=LogisticRegression.stocGradAscent1(ds,150);
		for(double weight:weights){
			System.out.print(weight+"\t");
		}
		System.out.println();		
		
		LogisticRegression.multiTest();
	}
	

	
 
	
	public static double[] stocGradAscent1(DoubleArrayDataSet ds,int numIter){
		int m=ds.getRowNumber();
		int n=ds.getColNumber();
	
		
		double[] weights=new double[n];
		Arrays.fill(weights, 1);
		Random rnd=new Random();
		for(int k=0;k<numIter;k++){
			ArrayList<Integer> allIndices=new ArrayList<Integer>(m);
			for(int i=0;i<m;i++){
				allIndices.add(i);
			}
			for(int i=0;i<m;i++){
				double alpha=4.0/(1.0+k+i)+0.01;
				int rndIdx=rnd.nextInt(allIndices.size());
				int idx=allIndices.get(rndIdx);
				allIndices.remove(rndIdx);
				
				double h=MathTools.sigmoid(MathTools.dotProduct(ds.data[idx], weights));
				double error=Double.valueOf(ds.labels[idx])-h;
				for(int j=0;j<weights.length;j++){
					weights[j]=weights[j]+alpha*error*ds.data[idx][j];
				}
			}
		}
		return weights;		
	}
	
	public static void multiTest() throws IOException{
		int numTest=10;
		double errorSum=0;
		for(int i=0;i<numTest;i++){
			errorSum+=colicTest();
		}
		System.out.println("After "+numTest+" Iterator average Error ratio is: "+(errorSum/numTest));
	}
	
	public static double colicTest() throws IOException{
		List<String> lines=FileTools.readFile2List("src/main/resources/ch05/horseColicTraining.txt");
		double[][] trainingData=new double[lines.size()][];
		String[] trainingLabels=new String[lines.size()];
		int i=0;
		for(String line:lines){
			String[] array=line.split("\t");
			double[] fea=new double[array.length-1];
			for(int j=0;j<fea.length;j++){
				fea[j]=Double.valueOf(array[j]);
			}
			trainingData[i]=fea;
			trainingLabels[i]=array[array.length-1];
			i++;
		}
		
		double[] weights=stocGradAscent1(new DoubleArrayDataSet(trainingData, trainingLabels), 500);
		int errorCount=0;
		int testCount=0;
		lines=FileTools.readFile2List("src/main/resources/ch05/horseColicTest.txt");
		for(String line:lines){
			testCount++;
			String[] array=line.split("\t");
			int label=(int)Double.valueOf(array[array.length-1]).doubleValue();
			double[] vec=new double[array.length-1];
			for(int j=0;j<vec.length;j++){
				vec[j]=Double.valueOf(array[j]);
			}
			int predictedLabel=classify(weights, vec);
			if(label!=predictedLabel){
				errorCount++;
			}
		}
		
		return 1.0*errorCount/testCount;
	}
	
	public static int classify(double[] weights,double[] vec){
		double prob=MathTools.sigmoid(MathTools.dotProduct(weights, vec));
		
		if(prob>0.5) return 1;
		else return 0;
	}
	
	public static double[] stocGradAscent0(DoubleArrayDataSet ds){
		int m=ds.getRowNumber();
		int n=ds.getColNumber();
		double alpha=0.01;
		
		double[] weights=new double[n];
		Arrays.fill(weights, 1);
		for(int i=0;i<m;i++){
			double h=MathTools.sigmoid(MathTools.dotProduct(ds.data[i], weights));
			double error=Double.valueOf(ds.labels[i])-h;
			for(int j=0;j<weights.length;j++){
				weights[j]=weights[j]+alpha*error*ds.data[i][j];
			}
		}
		
		return weights;
	}
	
	public static double[] gradAscent(DoubleArrayDataSet ds){
		int m=ds.getRowNumber();
		int n=ds.getColNumber();
		double alpha=0.001;
		int maxCycles=500;
		double[] weights=new double[n];
		Arrays.fill(weights, 1);
		double[] labelVec=new double[m];
		for(int i=0;i<m;i++){
			labelVec[i]=Double.valueOf(ds.labels[i]);
		}
		double[][] transposedMatrix=MathTools.transposeMatrix(ds.data);
		for(int i=0;i<maxCycles;i++){
			double[] h=MathTools.maxtrixMultiply(ds.data, weights);
			for(int j=0;j<h.length;j++){
				h[j]=MathTools.sigmoid(h[j]);
			}
			double[] error=MathTools.vecSubtraction(labelVec, h);
			double[] vec=MathTools.maxtrixMultiply(transposedMatrix, error);
			for(int j=0;j<weights.length;j++){
				weights[j]=weights[j]+alpha*vec[j];
			}
		}
		
		return weights;
	}
	
	public static DoubleArrayDataSet loadDataSet() throws IOException{
		List<String> lines=FileTools.readFile2List("src/main/resources/ch05/testSet.txt");
		double[][] data=new double[lines.size()][];
		String[] labels=new String[lines.size()];
		int lineNum=0;
		for(String line:lines){
			String[] array=line.split("\t");
			labels[lineNum]=array[2];
			double[] row=new double[]{
				1.0,
				Double.valueOf(array[0]),
				Double.valueOf(array[1])
			};
			data[lineNum]=row;
			lineNum++;
		}
		
		return new DoubleArrayDataSet(data, labels);
	}

}
