package org.mlia.utils;

public class ObjectArrayDataSet{
	public String[] labels;
	public Object[][] data;
	
	public int getRowNumber(){
		return data.length;
	}
	public int getColNumber(){
		return data[0].length;
	}
	
	public ObjectArrayDataSet(){
		
	}
	
	public ObjectArrayDataSet(Object[][] data,String[] labels){
		this.data=data;
		this.labels=labels;
	}
	
	public Object[] getCol(int index){
		Object[] col=new Object[data.length];
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
		for(Object[] row:data){
			for(Object cell:row){
				sb.append("\t").append(cell);
			}
			sb.append("\n");
		}
		
		
		return sb.toString();
	}
}
