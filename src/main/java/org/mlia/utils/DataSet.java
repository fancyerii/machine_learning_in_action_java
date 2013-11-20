package org.mlia.utils;

public class DataSet{
	public String[] labels;
	public double[][] data;
	
	public int getRowNumber(){
		return data.length;
	}
	public int getColNumber(){
		return data[0].length;
	}
	
	public double[] getCol(int index){
		double[] col=new double[data.length];
		for(int i=0;i<data.length;i++){
			col[i]=data[i][index];
		}
		
		return col;
	}
	
	@Override
	public String toString(){
		StringBuilder sb=new StringBuilder();
		sb.append("labels\n");
		for(String label:labels){
			sb.append("\t").append(label);
		}
		sb.append("\n");
		sb.append("data\n");
		for(double[] row:data){
			for(double cell:row){
				sb.append("\t").append(cell);
			}
			sb.append("\n");
		}
		
		
		return sb.toString();
	}
}
