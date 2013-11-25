package org.mlia.ch04;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.log4j.Logger;
import org.mlia.utils.FileTools;
import org.mlia.utils.ObjectArrayDataSet;

public class NaiveBayes {
	protected static Logger logger=Logger.getLogger(NaiveBayes.class);
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		ObjectArrayDataSet dataSet=loadData();
		System.out.println(dataSet);
		Map<String,Integer> wordVec=createVocabList(dataSet);
		BitSet bs=setOfWords2Vec(wordVec, dataSet.data[0]);
		System.out.println(bs);
		bs=setOfWords2Vec(wordVec, dataSet.data[5]);
		System.out.println(bs);
		
		NBModelParams nbModelParams=trainNB0(dataSet);
		System.out.println(nbModelParams);
		
		testNB();
		spamTest();
	}
	
	public static void testNB(){
		ObjectArrayDataSet dataSet=loadData();
		NBModelParams nbModelParams=trainNB0(dataSet);
		String[] testEntry =new String[] {"love", "my", "dalmation"};
		System.out.println("classified as "+classifyNB0(nbModelParams, testEntry));
		
		testEntry=new String[]{"stupid", "garbage"};
		System.out.println("classified as "+classifyNB0(nbModelParams, testEntry));
				
	}
	
	public static NBModelParams trainNB(ObjectArrayDataSet ds){
		Map<String,Integer> wordVec=createVocabList(ds);
		double pAbusive=0;
		int numTrainDocs=ds.getRowNumber();
		int numWords=wordVec.size();
		int abusiveCount=0;
		for(String label:ds.labels){
			if(label.equals("1")){
				abusiveCount++;
			}
		}
		pAbusive=1.0*abusiveCount/numTrainDocs;
		int p0Denom=2;//should be numWords? to normalize as a probability
		int p1Denom=2;//should be numWords? to normalize as a probability
		
		int[] p0Vec=new int[numWords];
		int[] p1Vec=new int[numWords];
		Arrays.fill(p0Vec, 1);
		Arrays.fill(p1Vec, 1);
		
		
		for(int i=0;i<numTrainDocs;i++){
			String label=ds.labels[i];
			Object[] row=ds.data[i];
			Map<Integer,Integer> bagOfWords=bagOfWords2Vec(wordVec, row);
			if(label.equals("1")){
				for(Entry<Integer,Integer> entry:bagOfWords.entrySet()){
					p1Vec[entry.getKey()]+=entry.getValue();
					p1Denom+=entry.getValue();
				}
			}else{
				for(Entry<Integer,Integer> entry:bagOfWords.entrySet()){
					p0Vec[entry.getKey()]+=entry.getValue();
					p0Denom+=entry.getValue();
				}				
			}
		}
		double[][] likelihood=new double[2][];
		likelihood[0]=new double[numWords];
		for(int i=0;i<numWords;i++){
			likelihood[0][i]=Math.log(1.0*p0Vec[i]/p0Denom);
		}
		likelihood[1]=new double[numWords];
		for(int i=0;i<numWords;i++){
			likelihood[1][i]=Math.log(1.0*p1Vec[i]/p1Denom);
		}
		double[] prior=new double[]{1-pAbusive,pAbusive};
		
		return new NBModelParams(likelihood, prior, wordVec);
	}
	
	
	public static NBModelParams trainNB0(ObjectArrayDataSet ds){
		Map<String,Integer> wordVec=createVocabList(ds);
		double pAbusive=0;
		int numTrainDocs=ds.getRowNumber();
		int numWords=wordVec.size();
		int abusiveCount=0;
		for(String label:ds.labels){
			if(label.equals("1")){
				abusiveCount++;
			}
		}
		pAbusive=1.0*abusiveCount/numTrainDocs;
		int p0Denom=2;//should be numWords? to normalize as a probability
		int p1Denom=2;//should be numWords? to normalize as a probability
		
		int[] p0Vec=new int[numWords];
		int[] p1Vec=new int[numWords];
		Arrays.fill(p0Vec, 1);
		Arrays.fill(p1Vec, 1);
		
		
		for(int i=0;i<numTrainDocs;i++){
			String label=ds.labels[i];
			Object[] row=ds.data[i];
			BitSet bs=setOfWords2Vec(wordVec, row);
			if(label.equals("1")){
				for(int j=0;j<bs.length();j++){
					if(bs.get(j)){
						p1Vec[j]++;
						p1Denom++;
					}
				}
			}else{
				for(int j=0;j<bs.length();j++){
					if(bs.get(j)){
						p0Vec[j]++;
						p0Denom++;
					}
				}				
			}
		}
		double[][] likelihood=new double[2][];
		likelihood[0]=new double[numWords];
		for(int i=0;i<numWords;i++){
			likelihood[0][i]=Math.log(1.0*p0Vec[i]/p0Denom);
		}
		likelihood[1]=new double[numWords];
		for(int i=0;i<numWords;i++){
			likelihood[1][i]=Math.log(1.0*p1Vec[i]/p1Denom);
		}
		double[] prior=new double[]{1-pAbusive,pAbusive};
		
		return new NBModelParams(likelihood, prior, wordVec);
	}
	
	private static String[] textParse(String text){
		Pattern pattern=Pattern.compile("\\w+");
		Matcher matcher=pattern.matcher(text);
		ArrayList<String> list=new ArrayList<String>();
		while(matcher.find()){
			list.add(matcher.group(0));
		}
		String[] result=new String[list.size()];
		for(int i=0;i<result.length;i++){
			result[i]=list.get(i);
		}
		
		return result;
	}
	
	public static void spamTest() throws IOException{
		
		ArrayList<String[]> allData=new ArrayList<String[]>();
		ArrayList<String> labels=new ArrayList<String>();
		for(File file:new File("src/main/resources/ch04/email/ham").listFiles()){
			List<String> lines=FileTools.readFile2List(file.getAbsolutePath());
			StringBuilder sb=new StringBuilder();
			for(String line:lines){
				sb.append(line).append("\n");
			}
			allData.add(textParse(sb.toString()));
			labels.add("0");
		}
		for(File file:new File("src/main/resources/ch04/email/spam").listFiles()){
			List<String> lines=FileTools.readFile2List(file.getAbsolutePath());
			StringBuilder sb=new StringBuilder();
			for(String line:lines){
				sb.append(line).append("\n");
			}
			allData.add(textParse(sb.toString()));
			labels.add("1");
		}		
		
		
		int testSize=30;
		Object[][] trainingData=new Object[allData.size()-testSize][];
		String[] trainingLabels=new String[trainingData.length];
		String[][] testData=new String[testSize][];
		String[] testLabels=new String[testSize];
		Set<Integer> testDataIdx=new HashSet<Integer>();
		Random rnd=new Random();
		
		while(testDataIdx.size()<testSize){
			int idx=rnd.nextInt(allData.size());
			testDataIdx.add(idx);
		}
		
		int trainIdx=0;
		int testIdx=0;
		for(int i=0;i<allData.size();i++){
			if(testDataIdx.contains(i)){
				testData[testIdx]=allData.get(i);
				testLabels[testIdx]=labels.get(i);
				testIdx++;
			}else{
				trainingData[trainIdx]= allData.get(i);
				trainingLabels[trainIdx]= labels.get(i) ;
				trainIdx++;
			}
		}
		
		NBModelParams nbModel=trainNB(new ObjectArrayDataSet(trainingData, trainingLabels));
		int errorCount=0;
		
		for(int i=0;i<testSize;i++){
			String predictedClass=classifyNB(nbModel, testData[i])+"";
			if(!predictedClass.equals(testLabels[i])){
				errorCount++;
				System.out.print("error[real="+testLabels[i]+",classified="+predictedClass+"]: ");
				for(String word:testData[i]){
					System.out.print("\t"+word);
				}
				System.out.println();
			}
		}
		
		System.out.println("errorCount="+errorCount+", errorRatio="+(1.0*errorCount/testSize));
	}
	

	public static int classifyNB(NBModelParams nbModel,String[] text){
		Map<Integer,Integer> bagOfWords=bagOfWords2Vec(nbModel.wordVec, text);
		double[] posteriors=new double[nbModel.prior.length];
		for(int i=0;i<posteriors.length;i++){
			double prob=nbModel.prior[i];
			for(Entry<Integer,Integer> entry:bagOfWords.entrySet()){
				prob+=nbModel.likelihood[i][entry.getKey()]*entry.getValue();
			}
			posteriors[i]=prob;
		}
		
		int bestClass=0;
		double bestProb=posteriors[0];
		for(int i=1;i<posteriors.length;i++){
			if(posteriors[i]>bestProb){
				bestClass=i;
				bestProb=posteriors[i];
			}
		}
		return bestClass;
	}
	
	public static int classifyNB0(NBModelParams nbModel,String[] text){
		BitSet bs=setOfWords2Vec(nbModel.wordVec, text);
		double[] posteriors=new double[nbModel.prior.length];
		for(int i=0;i<posteriors.length;i++){
			double prob=nbModel.prior[i];
			for(int j=0;j<bs.length();j++){
				if(bs.get(j)){
					prob+=nbModel.likelihood[i][j];
				}
			}
			posteriors[i]=prob;
		}
		
		int bestClass=0;
		double bestProb=posteriors[0];
		for(int i=1;i<posteriors.length;i++){
			if(posteriors[i]>bestProb){
				bestClass=i;
				bestProb=posteriors[i];
			}
		}
		return bestClass;
	}
	
	public static Map<String,Integer> createVocabList(ObjectArrayDataSet dataSet){
 
		Object[][] data=dataSet.data;

		Map<String,Integer> wordVec=new HashMap<String,Integer>();
		int idx=0;
		for(Object[] row:data){
			for(Object col:row){
				String word=(String)col;
				if(!wordVec.containsKey(word)){
					wordVec.put(word, idx++);
				}
			}
		}
		return wordVec;
	}
	
	public static BitSet setOfWords2Vec(Map<String,Integer> wordVec, Object[] inputSet){
		BitSet bs=new BitSet();
		for(Object word:inputSet){
			Integer idx=wordVec.get(word);
			if(idx==null){//unknown words,maybe not in training set
				logger.warn("unknown word: "+word);
			}else{
				bs.set(idx);
			}
		}
		
		return bs;
	}
	
	public static Map<Integer,Integer> bagOfWords2Vec(Map<String,Integer> wordVec,Object[] inputSet){
		Map<Integer,Integer> counter=new HashMap<Integer, Integer>();
		for(Object word:inputSet){
			Integer idx=wordVec.get(word);
			if(idx==null){//unknown words,maybe not in training set
				logger.warn("unknown word: "+word);
			}else{
				Integer count=counter.get(idx);
				if(count==null){
					counter.put(idx, 1);
				}else{
					counter.put(idx, 1+count);
				}
			}
		}		
		
		return counter;
	}
	
	public static ObjectArrayDataSet loadData(){
		Object[][] data={
			{"my", "dog", "has", "flea", "problems", "help", "please"},
			{"maybe", "not", "take", "him", "to", "dog", "park", "stupid"},
			{"my", "dalmation", "is", "so", "cute", "I", "love", "him"},
			{"stop", "posting", "stupid", "worthless", "garbage"},
			{"mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him"},
			{"quit", "buying", "worthless", "dog", "food", "stupid"}
      	};
		String[] labels=new String[]{
			"0",
			"1",
			"0",
			"1",
			"0",
			"1"
		};
		
		return new ObjectArrayDataSet(data, labels);
	}

}

class NBModelParams{
	public double[][] likelihood;
	public double[] prior;
	public Map<String,Integer> wordVec;
	
	public NBModelParams(double[][] likelihood, double[] prior,Map<String,Integer> wordVec){
		if(likelihood.length!=prior.length){
			throw new IllegalArgumentException("likelihood's length("+likelihood.length+") should be the same as prior ("+prior.length+")");
		}
		this.likelihood=likelihood;
		this.prior=prior;
		this.wordVec=wordVec;
	}
	
	@Override 
	public String toString(){
		StringBuilder sb=new StringBuilder();
		for(int i=0;i<likelihood.length;i++){
			sb.append("Likelihood of class "+(i+1)+": ");
			double[] row=likelihood[i];
			for(double col:row){
				sb.append(col).append("\t");
			}
			sb.setCharAt(sb.length()-1, '\n');
		}
		for(int i=0;i<prior.length;i++){
			sb.append("Prior of class "+(i+1)+": ");
			sb.append(prior[i]).append("\n");
		}
		return sb.toString();
	}
}
