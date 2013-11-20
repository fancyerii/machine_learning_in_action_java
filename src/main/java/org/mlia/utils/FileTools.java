package org.mlia.utils;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
 
public class FileTools {
	protected static Logger logger = Logger.getLogger(FileTools.class);
	
	public static List<String> readFile2List(String filePath) throws IOException{
		return readFile2List(filePath, Charset.defaultCharset().name());
	}
	
	public static List<String> readFile2List(InputStream is) throws IOException{
		return readFile2List(is, Charset.defaultCharset().name());
	}
	
	public static List<String> readFile2List(InputStream is,String encoding) throws IOException{
		List<String> lines=new ArrayList<String>();
		BufferedReader br=null;
		
		try {
			br=new BufferedReader(new InputStreamReader(is,encoding));
			String line;
			while((line=br.readLine())!=null){
				lines.add(line);
			}
		}finally{
			try {
				if(br!=null) br.close();
			} catch (IOException e) {
				logger.error(e.getMessage(), e);
			}
			
		}		
		return lines;
	}
	
	public static List<String> readFile2List(String filePath,String encoding) throws IOException{
		return readFile2List(new FileInputStream(filePath), encoding);
	}
		
}
