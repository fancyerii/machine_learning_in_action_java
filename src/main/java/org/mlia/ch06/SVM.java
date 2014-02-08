package org.mlia.ch06;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.mlia.utils.DoubleArrayDataSet;
import org.mlia.utils.FileTools;
import org.mlia.utils.MathTools;

public class SVM {
	public static void main(String[] args) throws IOException{
		DoubleArrayDataSet ds=SVM.loadDataSet("src/main/resources/ch06/testSet.txt");
		//System.out.println(ds);
		//AlphasAndB params=SVM.smoSimple(ds.data, convert(ds.labels), 0.6, 0.001, 40);
		//System.out.println(params);
		AlphasAndB params=SVM.smoP(ds.data, convert(ds.labels), 0.6, 0.001, 40);
		System.out.println(params);		
	}
	
	private static double[] convert(String[] arr){
		double[] result=new double[arr.length];
		for(int i=0;i<arr.length;i++){
			result[i]=Double.valueOf(arr[i]);
		}
		
		return result;
	}
	
	public static DoubleArrayDataSet loadDataSet(String fileName) throws IOException{
		DoubleArrayDataSet dataSet=new DoubleArrayDataSet();
		List<String> lines=FileTools.readFile2List(fileName, "UTF-8");
		double[][] data=new double[lines.size()][];
		String[] labels=new String[lines.size()];
		int lineNum=0;
		for(String line:lines){
			String[] arr=line.split("\t");
			data[lineNum]=new double[]{Double.valueOf(arr[0]), Double.valueOf(arr[1])}; 
			labels[lineNum]=arr[2];
			lineNum++;
		}
		dataSet.data=data;
		dataSet.labels=labels;
		return dataSet;
	}

	private static int selectJrand(int i, int m, Random rnd){
		int j=i;
		while(j==i){
			j=rnd.nextInt(m);
		}
		return j;
	}
	
	private static double clipAlpha(double aj, double H, double L){
		if(aj>H) return H;
		if(aj<L) return L;
		return aj;
	}

	
	private static double fx(int i, double[] alphas, double b, double[][] data, double[] labels){
		double f=b;
		for(int j=0;j<data.length;j++){
			f+=(alphas[j]*labels[j]*(MathTools.dotProduct(data[i], data[j])));
		}
		return f;
	}
	
	private static double calcEk(optStruct os, int k){
		double fxk=fx(k, os.alphas, os.b, os.data, os.labels);
		return fxk-os.labels[k];
	}
	
	private static List<Integer> nonZero(double[][] cache){
		List<Integer> result=new ArrayList<Integer>();
		for(int i=0;i<cache.length;i++){
			double[] row=cache[i];
			if(row[0]!=0){
				result.add(i);
			}
		}
		return result;
	}
	
	private static List<Integer> nonZero(optStruct os){
		List<Integer> result=new ArrayList<Integer>();
		for(int i=0;i<os.alphas.length;i++){
			if(os.alphas[i]>0 && os.alphas[i]<os.C){
				result.add(i);
			}
		}
		return result;
	}
	
	private static double[] selectJ(int i, optStruct os, double ei){
		int maxK=-1;
		double maxDeltaE=0;
		double ej=0;
		os.eCache[i][0]=1;
		os.eCache[i][1]=ei;
		List<Integer> validEcacheList=nonZero(os.eCache);
		if(validEcacheList.size()>1){
			for(int k:validEcacheList){
				if(k==i) continue;
				double ek=calcEk(os, k);
				double deltaE=Math.abs(ei-ek);
				if(deltaE>maxDeltaE || maxK==-1){//if maxK is -1 then 
					maxDeltaE=deltaE;
					maxK=k;
					ej=ek;
				}
			}
			return new double[]{maxK, ej}; 
		}else{
			int j=selectJrand(i, os.m, new Random());
			ej=calcEk(os, j);
			return new double[]{j, ej};
		}
	}
	
	private static void updateEk(optStruct os, int k){
		double ek=calcEk(os, k);
		os.eCache[k][0]=1;
		os.eCache[k][1]=ek;
	}
	
	private static int innerL(int i, optStruct os){
		double ei=calcEk(os, i);
		if((os.labels[i]*ei < -os.toler && os.alphas[i] < os.C)
				||(os.labels[i]*ei > os.toler && os.alphas[i] > 0)){
			double[] arr=selectJ(i, os, ei);
			int j=(int)arr[0];
			double ej=arr[1];
			double alphaIold=os.alphas[i];
			double alphaJold=os.alphas[j];
			double L=0;
			double H=0;
			if(os.labels[i]!=os.labels[j]){
				L=Math.max(0, os.alphas[j]-os.alphas[i]);
				H=Math.min(os.C, os.C+os.alphas[j]-os.alphas[i]);
			}else{
				L=Math.max(0, os.alphas[j]+os.alphas[i]-os.C);
				H=Math.min(os.C, os.alphas[j]+os.alphas[i]);
			}
			if(L==H){
				System.out.println("L==H");
				return 0;
			}
			double eta=2.0*MathTools.dotProduct(os.data[i], os.data[j])
					- MathTools.dotProduct(os.data[i], os.data[i])
					- MathTools.dotProduct(os.data[j], os.data[j]);
			
			if(eta>=0){
				System.out.println("eta>=0");
				return 0;
			}
			os.alphas[j] -= os.labels[j]*(ei-ej)/eta;
			os.alphas[j] = clipAlpha(os.alphas[j], H, L);
			updateEk(os, j);
			if(Math.abs(os.alphas[j]-alphaJold) < 0.00001){
				System.out.println("j not moving enough");
				return 0;
			}
			os.alphas[i] += os.labels[j]*os.labels[i]*(alphaJold-os.alphas[j]);
			updateEk(os, i);
			double b1=os.b-ei
					-os.labels[i]*(os.alphas[i]-alphaIold)*MathTools.dotProduct(os.data[i], os.data[i])
					-os.labels[j]*(os.alphas[j]-alphaJold)*MathTools.dotProduct(os.data[i], os.data[j]);
			double b2=os.b-ej
					-os.labels[i]*(os.alphas[i]-alphaIold)*MathTools.dotProduct(os.data[i], os.data[j])
					-os.labels[j]*(os.alphas[j]-alphaJold)*MathTools.dotProduct(os.data[j], os.data[j]);
			
			if(0 < os.alphas[i] && os.alphas[i] < os.C){
				os.b=b1;
			}else if(0 < os.alphas[j] && os.alphas[j] < os.C){
				os.b=b2;
			}else{
				os.b=(b1+b2)/2;
			}
			
			return 1;
		}else{
			return 0;
		}
	}
	
	public static AlphasAndB smoP(double[][]data, double[] labels, double C, double toler, int maxIter){
		optStruct os=new optStruct(data, labels, C, toler);
		int iter=0;
		boolean entireSet=true;
		int alphaPairsChanged=0;
		while (iter < maxIter && (alphaPairsChanged > 0 || entireSet)){
			alphaPairsChanged=0;
			if(entireSet){
				for(int i=0;i<os.m;i++){
					alphaPairsChanged+=innerL(i, os);
					System.out.println("fullSet, iter: "+iter+" i:"+i+", pairs changed "+alphaPairsChanged);
					iter++;
				}
			}else{
				List<Integer> nonBoundIs=nonZero(os);
				for(int i:nonBoundIs){
					alphaPairsChanged+=innerL(i, os);
					System.out.println("non-bound, iter: "+iter+" i:"+i+", pairs changed "+alphaPairsChanged);
					iter++;
				}
			}
			if(entireSet){
				entireSet=false;
			}else if(alphaPairsChanged==0){
				entireSet=true;
			}
			System.out.println("iteration number: "+iter);
		}
		
		return new AlphasAndB(os.alphas, os.b);
	}
	
	public static AlphasAndB smoSimple(double[][]data, double[] labels, double C, double toler, int maxIter){
		double b=0;
		int m=data.length;
		//int n=data[0].length;
		double[] alphas=new double[m];
		int iter=0;
		int alphaPairsChanged;
		Random rnd=new Random();
		while(iter<maxIter){
			alphaPairsChanged=0;
			for(int i=0;i<m;i++){
				double fxi=fx(i, alphas, b, data, labels);
				double ei=fxi-labels[i];
				if((labels[i]*ei < -toler && alphas[i] < C)
					||(labels[i]*ei > toler && alphas[i] > 0)){
					int j=selectJrand(i, m, rnd);
					double fxj=fx(j, alphas, b, data, labels);
					double ej=fxj-labels[j];
					double alphaIold=alphas[i];
					double alphaJold=alphas[j];
					double L=0;
					double H=0;
					if(labels[i]!=labels[j]){
						L=Math.max(0, alphas[j]-alphas[i]);
						H=Math.min(C, C+alphas[j]-alphas[i]);
					}else{
						L=Math.max(0, alphas[j]+alphas[i]-C);
						H=Math.min(C, alphas[j]+alphas[i]);
					}
					if(L==H){
						System.out.println("L==H");
						continue;
					}
					double eta=2.0*MathTools.dotProduct(data[i], data[j])
							- MathTools.dotProduct(data[i], data[i])
							- MathTools.dotProduct(data[j], data[j]);
					
					if(eta>=0){
						System.out.println("eta>=0");
						continue;
					}
					alphas[j] -= labels[j]*(ei-ej)/eta;
					alphas[j] = clipAlpha(alphas[j], H, L);
					if(Math.abs(alphas[j]-alphaJold) < 0.00001){
						System.out.println("j not moving enough");
						continue;
					}
					alphas[i] += labels[j]*labels[i]*(alphaJold-alphas[j]);
					double b1=b-ei
							-labels[i]*(alphas[i]-alphaIold)*MathTools.dotProduct(data[i], data[i])
							-labels[j]*(alphas[j]-alphaJold)*MathTools.dotProduct(data[i], data[j]);
					double b2=b-ej
							-labels[i]*(alphas[i]-alphaIold)*MathTools.dotProduct(data[i], data[j])
							-labels[j]*(alphas[j]-alphaJold)*MathTools.dotProduct(data[j], data[j]);
					
					if(0 < alphas[i] && alphas[i] < C){
						b=b1;
					}else if(0 < alphas[j] && alphas[j] < C){
						b=b2;
					}else{
						b=(b1+b2)/2;
					}
					alphaPairsChanged++;
					System.out.println("iter: "+iter+" i:"+i+", pairs changed "+alphaPairsChanged);
				}

			}
			if(alphaPairsChanged==0) iter++;
			else iter=0;
			System.out.println("iteration number: "+iter);
		}
		
		return new AlphasAndB(alphas, b);
	}
}

class AlphasAndB{
	public double[] alphas;
	public double b;
	public AlphasAndB(double[] alphas, double b){
		this.b=b;
		this.alphas=alphas;
	}
	
	@Override
	public String toString(){
		StringBuilder sb=new StringBuilder();
		sb.append("b: ").append(b).append("\n");
		sb.append("alphas: ");
		for(double alpha:this.alphas){
			sb.append(alpha).append("\t");
		}
		sb.append("\n");
		return sb.toString();
	}
}

class optStruct{
	public double[][] data;
	public double[] labels;
	public double C;
	public double toler;
	public double[] alphas;
	public double b;
	public double[][] eCache;
	public int m;
	
	public optStruct(double[][] data, double[] labels, double C, double toler){
		this.data=data;
		this.labels=labels;
		this.C=C;
		this.toler=toler;
		this.m=data.length;
		this.alphas=new double[m];
		this.b=0;
		this.eCache=new double[m][];
		for(int i=0;i<m;i++){
			this.eCache[i]=new double[2];
		}
	}
}
