package org.mlia.ch03;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

 
import org.mlia.utils.FileTools;
import org.mlia.utils.ObjectArrayDataSet;

public class DecisionTree {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		ObjectArrayDataSet ds=DecisionTree.createDataSet();
		System.out.println(ds);
		double ent=DecisionTree.calcShannonEnt(ds);
		System.out.println(ent);
		
		
		ObjectArrayDataSet split1=DecisionTree.splitDataSet(ds, 0, 1);
		System.out.println(split1);
		ObjectArrayDataSet split2=DecisionTree.splitDataSet(ds, 0, 0);
		System.out.println(split2);
		
		int bestFeature=DecisionTree.chooseBestFeatureToSplit(ds);
		System.out.println(bestFeature);
		String[] featureNames=new String[]{
			"no surfacing",
			"flippers"
		};
		
		
		
		DecisionTreeNode tree=DecisionTree.createTree(ds, featureNames);
		System.out.println(tree.printTree());
		
		System.out.println(DecisionTree.classify(tree, featureNames, new Object[]{1,0}));
		
		System.out.println(DecisionTree.classify(tree, featureNames, new Object[]{1,1}));
		
		System.out.println(DecisionTree.classify(tree, featureNames, new Object[]{0,1}));
		
		System.out.println(DecisionTree.classify(tree, featureNames, new Object[]{0,0}));
 
		DecisionTree.classifyLenses();
	}
	
	
	public static void classifyLenses() throws Exception{
		ObjectArrayDataSet ds=readLensesData();
		String[] featureNames=new String[]{
			"age",
			"prescript",
			"astigmatic",
			"tearRate"
		};
		DecisionTreeNode tree=DecisionTree.createTree(ds, featureNames);
		System.out.println(tree.printTree());
		
		
	}
	
	private static ObjectArrayDataSet readLensesData() throws Exception{
		String filePath="src/main/resources/ch03/lenses.txt";
		List<String> lines=FileTools.readFile2List(filePath);

		Object[][] data=new Object[lines.size()][];
		String[] labels=new String[lines.size()];
		
		int i=0;
		for(Iterator<String> iter=lines.iterator();i<lines.size();i++){
			String line=iter.next();
			String[] array=line.split("\t");
			
			 
			
			data[i]=new Object[]{array[0],array[1],array[2],array[3]};
			labels[i]=array[4];
			
		}
		
		return new ObjectArrayDataSet(data,labels);
		
	}
	
	private static boolean onlyOneClass(String[] classList){
		String cls=classList[0];
		for(int i=1;i<classList.length;i++){
			if(!cls.equals(classList[i])){
				return false;
			}
		}
		
		return true;
	}
	
	public static DecisionTreeNode createTree(ObjectArrayDataSet ds,String[] featureNames){
		String[] classList=ds.labels;
		if(onlyOneClass(classList)){// stop when all classes are the same.
			DecisionTreeNode node=new DecisionTreeNode();
			node.children.add(classList[0]);
			return node;
		}
		if(ds.getColNumber()==0){
			DecisionTreeNode node=new DecisionTreeNode();
			node.children.add(majorityCnt(classList));
			return node;			
		}
		int bestFeature=chooseBestFeatureToSplit(ds);
		String bestFeatureLabel=featureNames[bestFeature];
		DecisionTreeNode node=new DecisionTreeNode();
		node.featureIdx=bestFeature;
		node.featureName=bestFeatureLabel;
		String[] subfeatureNames=new String[featureNames.length-1];
		for(int i=0;i<bestFeature;i++){
			subfeatureNames[i]=featureNames[i];
		}
		for(int i=bestFeature+1;i<featureNames.length;i++){
			subfeatureNames[i-1]=featureNames[i];
		}
		Object[] featValues=ds.getCol(bestFeature);
		Set<Object> uniqueVals=getUniqFeatures(featValues);
		for(Object value:uniqueVals){
			
			
			DecisionTreeNode child=createTree(splitDataSet(ds, bestFeature, value), subfeatureNames);
			node.featureValues.add(value);
			if(child.children.size()==1){//leaf
				node.children.add(child.children.get(0));
			}else{
				node.children.add(child);
			}
		}
		
		return node;
	}
	
	public static String majorityCnt(String[] classList){
		Map<String,Integer> classCount=new HashMap<String,Integer>();
		for(String cls:classList){
			if(classCount.containsKey(cls)){
				classCount.put(cls, 1+classCount.get(cls));
			}else{
				classCount.put(cls, 1);
			}
		}
		
		int maxCount=0;
		String maxCountClass=null;
		for(Entry<String,Integer> entry:classCount.entrySet()){
			if(entry.getValue()>maxCount){
				maxCount=entry.getValue();
				maxCountClass=entry.getKey();
			}
		}
		
		return maxCountClass;
	}
	
	private static Set<Object> getUniqFeatures(Object[] array){
		Set<Object> set=new HashSet<Object>();
		for(Object d:array){
			set.add(d);
		}
		
		return set;
	}
 
	public static int chooseBestFeatureToSplit(ObjectArrayDataSet ds){
		int numFeatures=ds.getColNumber();
		double baseEntropy=calcShannonEnt(ds);
		double bestInfoGain=0;
		int bestFeature=-1;
		for(int i=0;i<numFeatures;i++){
			Object[] featList=ds.getCol(i);
			Set<Object> uniqueVals=getUniqFeatures(featList);
			double newEntropy=0;
			for(Object value:uniqueVals){
				ObjectArrayDataSet subDataSet = splitDataSet(ds, i, value);
				double prob = subDataSet.getRowNumber()*1.0/ds.getRowNumber();
			    newEntropy += prob * calcShannonEnt(subDataSet);
			}
	        double infoGain = baseEntropy - newEntropy;
	        if (infoGain > bestInfoGain){
	            bestInfoGain = infoGain;
	            bestFeature = i;
	        }
		}
		
		return bestFeature;
		
	}
	
	public static ObjectArrayDataSet splitDataSet(ObjectArrayDataSet ds, int axis, Object value){
		ObjectArrayDataSet split=new ObjectArrayDataSet();
		ArrayList<Object[]> data=new ArrayList<Object[]>();
		ArrayList<String> labels=new ArrayList<String>();
		for(int i=0;i<ds.getRowNumber();i++){
			Object[] row=ds.data[i];
			if(row[axis].equals(value)){
				Object[] newRow=copyArrayExceptionOneElem(row, axis);
				
				data.add(newRow);
				labels.add(ds.labels[i]);
			}
		}
		
		
		split.data=new Object[data.size()][];
		split.labels=new String[data.size()];
		for(int i=0;i<split.data.length;i++){
			split.data[i]=data.get(i);
			split.labels[i]=labels.get(i);
		}
		return split;
	}
	
	public static ObjectArrayDataSet createDataSet(){
		ObjectArrayDataSet ds=new ObjectArrayDataSet();
		Object[][] data=new Object[][]{
			{1, 1},
			{1, 1},
			{1, 0},
			{0, 1},
			{0, 1}
		};
		ds.data=data;
		ds.labels=new String[]{
			"yes",
			"yes",
			"no",
			"no",
			"no"
		};
		return ds;
	}
	
	private static Object[] copyArrayExceptionOneElem(Object[] array, int index){
		Object[] newArray=new Object[array.length-1];
		for(int i=0;i<index;i++){
			newArray[i]=array[i];
		}
		for(int i=index+1;i<array.length;i++){
			newArray[i-1]=array[i];
		}
		return newArray;
	}
	
	private static String[] copyArrayExceptionOneElem(String[] array, int index){
		String[] newArray=new String[array.length-1];
		for(int i=0;i<index;i++){
			newArray[i]=array[i];
		}
		for(int i=index+1;i<array.length;i++){
			newArray[i-1]=array[i];
		}
		return newArray;		
	}
	
	public static String classify(DecisionTreeNode tree,String[] featureNames, Object[] testVec){

		Object feature=testVec[tree.featureIdx];
		String[] subFeatureNames=copyArrayExceptionOneElem(featureNames, tree.featureIdx);
		Object[] subVec=copyArrayExceptionOneElem(testVec, tree.featureIdx);
		for(int i=0;i<tree.children.size();i++){
			Object fv=tree.featureValues.get(i);
			if(fv.equals(feature)){
				Object child=tree.children.get(i);
				if(child instanceof String){
					return (String)child;
				}else{
					return classify((DecisionTreeNode) tree.children.get(i),subFeatureNames,subVec);
				}
			}
		}
		
		return null;
	}
	
	public static double calcShannonEnt(ObjectArrayDataSet ds){
		int numEntries=ds.getRowNumber();
		Map<String,Integer> labelCounts=new HashMap<String,Integer>();
		for(int i=0;i<numEntries;i++){
			String currentLabel=ds.labels[i];
			if(labelCounts.containsKey(currentLabel)){
				labelCounts.put(currentLabel, labelCounts.get(currentLabel)+1);
			}else{
				labelCounts.put(currentLabel, 1);
			}
		}
		double shannonEnt=0;
		for(Entry<String,Integer> entry:labelCounts.entrySet()){
			double prob=entry.getValue()*1.0/numEntries;
			shannonEnt-=prob*Math.log(prob)/Math.log(2);
		}
		return shannonEnt;
	}
}

class DecisionTreeNode implements Serializable{
	private static final long serialVersionUID = -910563028108144485L;
	public int featureIdx;
	public String featureName;
	public ArrayList<Object> children=new ArrayList<Object>();
	public ArrayList<Object> featureValues=new ArrayList<Object>();
	
	public String printTree(){
		StringBuilder sb=new StringBuilder();
		recursivePrint(this, sb, 0);
		return sb.toString();
	}
	
	private static void printTab(int count,StringBuilder sb){
		for(int i=0;i<count;i++){
			sb.append("\t");
		}
	}
	
	private static void recursivePrint(DecisionTreeNode curNode, StringBuilder sb, int depth){
 
		for(int i=0;i<curNode.children.size();i++){
			Object fv=curNode.featureValues.get(i);
			Object child=curNode.children.get(i);
			printTab(depth,sb);
			if(child instanceof String){
				sb.append("if (["+curNode.featureName+"]=='"+fv+"') then classify it as {").append((String)child).append("}\n");
			}else{
				sb.append("if (["+curNode.featureName+"]=='"+fv+"') then").append("\n");
				recursivePrint((DecisionTreeNode)child, sb, depth+1);
			}
			 
		}		
	}
	
	@Override
	public String toString(){
		StringBuilder sb=new StringBuilder();
		
		sb.append(featureName).append("\n");
		for(int i=0;i<children.size();i++){
			Object fv=featureValues.get(i);
			Object child=children.get(i);
			sb.append("\t");
			if(child instanceof String){
				sb.append("[if v=='"+fv+"'] classify as ").append((String)child);
			}else{
				sb.append("[if v=='"+fv+"'] need check other features");
			}
			sb.append("\n");
		}
		
		
		return sb.toString();
		
	}
}
