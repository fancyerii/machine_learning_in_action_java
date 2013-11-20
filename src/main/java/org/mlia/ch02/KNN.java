package org.mlia.ch02;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream; 
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.mlia.utils.DataSet;
import org.mlia.utils.FileTools;

 

public class KNN {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		DataSet ds=createDataSet();
		System.out.println(ds);
		double dist=calcEuclideanDistance(new double[]{1,0,0,1}, new double[]{7,6,9,4});
		System.out.println(dist);
		
		double[] dists=new double[]{1.0, 10.0, 5.0, 3.0, 1.0};
		int[] topIndices=getTopNIndices(dists, 3);
		for(int index:topIndices){
			if(index==-1){
				System.out.println("None");
			}else{
				System.out.println(dists[index]);
			}
		}
		
		double[] testData=new double[]{0.6,0.7};
		String label=KNN.classify0(testData, ds.data, ds.labels, 3);
		System.out.println("Point ("+testData[0]+","+testData[1]+") is classified as "+label);
		
		
		ds=KNN.file2Matrix("src/main/resources/ch02/datingTestSet2.txt");
		System.out.println(ds);
		int cols=ds.data[0].length;
		double[] minVals=new double[cols];
		double[] ranges=new double[cols];
		KNN.autoNorm(ds, minVals, ranges);
		System.out.println("After normalizing");
		System.out.println(ds);
		KNN.datingClassTest();
		
		//KNN.classifyPerson();
		
		double[] vec=KNN.img2Vector("src/main/resources/ch02/digits/trainingDigits/0_0.txt");
		for(int i=0;i<32;i++){
			for(int j=0;j<32;j++){
				if(vec[i*32+j]==0.0){
					System.out.print(" ");
				}else{
					System.out.print("1");
				}
			}
			System.out.println();
		} 
		
		KNN.handwritingClassTest();
	}

	/**
	 * x and y must have the same length
	 * @param x
	 * @param y
	 * @return
	 */
	private static double calcEuclideanDistance(double[] x, double[] y){
		double dist=0;
		for(int i=0;i<x.length;i++){
			dist+=(x[i]-y[i])*(x[i]-y[i]);
		}
		
		dist=Math.sqrt(dist);
		
		return dist;
	}
	
	/**
	 * use a priority queue to get top N small distances' indexes
	 * @param dists
	 * @param N
	 * @return
	 */
	private static int[] getTopNIndices(double[] dists,int N){
		int[] topIndices=new int[N];
		PriorityQueue<PriorityQueueElem> pq=new PriorityQueue<PriorityQueueElem>(N,new Comparator<PriorityQueueElem>() {
			
			@Override
			public int compare(PriorityQueueElem o1, PriorityQueueElem o2) {
				if(o1.data<o2.data) return 1;
				else return -1;
			}
		});
		 
		for(int i=0;i<dists.length;i++){
			PriorityQueueElem top=pq.peek();
			if(pq.size()<N) {
				pq.offer(new PriorityQueueElem(dists[i], i));
			}else{
				if(top.data<=dists[i]) continue;
				pq.remove();
				pq.offer(new PriorityQueueElem(dists[i], i));
			}
		}
		
		
		int lastPos=pq.size()-1;
		for(int i=lastPos+1;i<topIndices.length;i++){
			topIndices[i]=-1;
		}
		while(!pq.isEmpty()){
			topIndices[lastPos--]=pq.poll().index;
		}
		
		
		return topIndices;
	}
	
	public static String classify0(double[] inX,double[][] dataSet,String[] labels, int k){
		int dataSetSize=dataSet.length;
		double[] dists=new double[dataSetSize];
		for(int i=0;i<dataSetSize;i++){
			dists[i]=calcEuclideanDistance(inX, dataSet[i]);
		}
		int[] topIndices=getTopNIndices(dists, k);
		Map<String,Integer> counter=new HashMap<String,Integer>();
		for(int index:topIndices){
			Integer c=counter.get(labels[index]);
			if(c==null){
				counter.put(labels[index], 1);
			}else{
				counter.put(labels[index], 1+c);
			}
		}
		int maxCount=0;
		String maxLabel=null;
		for(Entry<String,Integer> entry:counter.entrySet()){
			if(entry.getValue()>maxCount){
				maxLabel=entry.getKey();
				maxCount=entry.getValue();
			}
		}
		return maxLabel;
	}
	
	public static double[] img2Vector(String filePath) throws IOException{
		double[] returnVect=new double[1024];
		List<String> lines=FileTools.readFile2List(filePath);
		Iterator<String> iter=lines.iterator();
		for(int i=0;i<32;i++){
			String line=iter.next();
			for(int j=0;j<32;j++){
				returnVect[32*i+j]=Double.valueOf(line.charAt(j)+"");
			}
		}
		return returnVect;
	}
	
	public static void handwritingClassTest() throws IOException{
		List<String> hwLabels=new ArrayList<String>();
		File dir=new File("src/main/resources/ch02/digits/trainingDigits");
		File[] files=dir.listFiles();
		int m=files.length;
		double[][] trainingMat=new double[m][];
		Pattern ptn=Pattern.compile("(\\d+)_(\\d+)\\.txt");
		for(int i=0;i<m;i++){
			trainingMat[i]=KNN.img2Vector(files[i].getAbsolutePath());
			String fn=files[i].getName();
			Matcher matcher=ptn.matcher(fn);
			if(!matcher.matches()){
				throw new RuntimeException("invalid file name: "+fn);
			}
			hwLabels.add(matcher.group(1));
		}
		String[] labels=hwLabels.toArray(new String[0]);
		int errorCount=0;
		dir=new File("src/main/resources/ch02/digits/testDigits");
		files=dir.listFiles();
		for(File file:files){
			double[] testVec=KNN.img2Vector(file.getAbsolutePath());
		 
			String fn=file.getName();
			
			Matcher matcher=ptn.matcher(fn);
			if(!matcher.matches()){
				throw new RuntimeException("invalid file name: "+fn);
			}
			String label=matcher.group(1);
			String classifierResult=KNN.classify0(testVec, trainingMat, labels, 3);
			
			if(!label.equals(classifierResult)){
				errorCount++;
				System.out.println("the classifer came back with: "+classifierResult
						+", the real answer is: "+label+" file: "+fn);
			}
		}
		
		System.out.println("total error count: "+errorCount);
		System.out.println("error rate: "+(1.0*errorCount/files.length));
		
		
	}
	
	public static DataSet file2Matrix(String filePath) throws IOException{
		return file2Matrix(new FileInputStream(filePath));
	}
	
	public static DataSet file2Matrix(InputStream is) throws IOException{
		DataSet dataSet=new DataSet(); 
		List<String> lines=FileTools.readFile2List(is);
		String[] labels=new String[lines.size()];
		double[][] data=new double[lines.size()][];
		dataSet.labels=labels;
		dataSet.data=data;
		
		int rowIndex=0;
		int cols=0;
		for(String line:lines){
			String[] array=line.split("\t");
			double[] row=new double[array.length-1];
			if(rowIndex==0){
				cols=row.length;
			}else if(cols!=row.length){
				throw new RuntimeException("line "+(rowIndex+1)+" has cols: "+row.length+" but first line has cols: "+cols);
			}
			for(int i=0;i<row.length;i++){
				row[i]=Double.valueOf(array[i]);
			}
			labels[rowIndex]=array[array.length-1];
			data[rowIndex]=row;
			rowIndex++;
		}
		
		return dataSet;
	}
	
	public static void autoNorm(DataSet dataSet,double[] minVals,double[] ranges){
		double[][] data=dataSet.data;
		int cols=data[0].length;
		for(int col=0;col<cols;col++){
			double min=data[0][col];
			double max=data[0][col];
			for(int row=1;row<data.length;row++){
				if(data[row][col]<min){
					min=data[row][col];
				}
				if(data[row][col]>max){
					max=data[row][col];
				}
			}
			minVals[col]=min;
			ranges[col]=max-min;
			
		}
		
		for(int row=0;row<data.length;row++){
			for(int col=0;col<cols;col++){
				data[row][col]=(data[row][col]-minVals[col])/ranges[col];
			}
		}
	}
	
	public static void classifyPerson() throws IOException{
		String[] resultList=new String[]{"not at all", "in small doses", "in large doses"};
		BufferedReader br=new BufferedReader(new InputStreamReader(System.in));
		System.out.println("percentage of time spent playing video games?");
		String line=br.readLine();
		double percentTats=Double.valueOf(line);
		System.out.println("frequent flier miles earned per year?");
		line=br.readLine();
		double ffMiles=Double.valueOf(line);
		System.out.println("liters of ice cream consumed per year?");
		line=br.readLine();
		double iceCream=Double.valueOf(line);
		DataSet ds=KNN.file2Matrix("src/main/resources/ch02/datingTestSet2.txt");
		int cols=ds.data[0].length;
		double[] minVals=new double[cols];
		double[] ranges=new double[cols];
		KNN.autoNorm(ds, minVals, ranges);
		double[] inArr=new double[]{
				(ffMiles-minVals[0])/ranges[0],
				(percentTats-minVals[1])/ranges[1],
				(iceCream-minVals[2])/ranges[2]
		};
		
		String classifierResult=KNN.classify0(inArr, ds.data, ds.labels, 3);
		System.out.println("You will probably like this person: "+resultList[Integer.valueOf(classifierResult)-1]);
	}
	
	public static void datingClassTest() throws IOException{
		double hoRatio=0.10;
		DataSet ds=KNN.file2Matrix("src/main/resources/ch02/datingTestSet2.txt");
		int cols=ds.data[0].length;
		double[] minVals=new double[cols];
		double[] ranges=new double[cols];
		autoNorm(ds, minVals, ranges);
		
		int numTestVecs=(int)(ds.data.length*hoRatio);
		int errorCount=0;
		double[][] trainingData=getTrainingData(ds.data, numTestVecs);
		String[] trainingLabels=getTrainingLabels(ds.labels, numTestVecs);
		for(int i=0;i<numTestVecs;i++){
			double[] row=ds.data[i];
			String classifierResult=classify0(row, trainingData, trainingLabels, 3);
			System.out.println("the classifier came back with "+classifierResult+
					", the real answer is: "+ds.labels[i]);
			
			if(!classifierResult.equals(ds.labels[i])){
				errorCount++;
			}
		}
		System.out.println("total error count: "+errorCount);
		System.out.println("the total error rate is: "+(1.0*errorCount/numTestVecs));
	}
	
	private static String[] getTrainingLabels(String[] labels,int startRow){
		String[] result=new String[labels.length-startRow];
		for(int i=0;i<result.length;i++){
			result[i]=labels[i+startRow];
		}
		
		return result;
	}
	
	private static double[][] getTrainingData(double[][] data,int startRow){
		double[][] result=new double[data.length-startRow][];
		for(int i=0;i<result.length;i++){
			result[i]=data[i+startRow];
		}
		
		return result;
	}
	
	public static DataSet createDataSet(){
		DataSet dataSet=new DataSet();
		String[] labels=new String[]{
			"A",
			"A",
			"B",
			"B"
		};
		dataSet.labels=labels;
		double[][] data=new double[][]{
			{1.0, 1.1},
			{1.0, 1.0},
			{0.0, 0.0},
			{0.0, 0.1}
		};
		dataSet.data=data;
		
		return dataSet;
	}
}

class PriorityQueueElem{
	public double data;
	public int index;
	public PriorityQueueElem(double data,int index){
		this.data=data;
		this.index=index;
	}
}
