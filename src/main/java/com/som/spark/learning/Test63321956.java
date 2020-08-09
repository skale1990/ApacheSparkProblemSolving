package com.som.spark.learning;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.util.ArrayList;
import java.util.List;

public class Test63321956 {
    public static void main(String[] args) {
        Logger.getLogger("org.apache").setLevel(Level.WARN);
        List<Integer> inputData = new ArrayList<>();

        inputData.add(25);


        SparkConf conf = new SparkConf().setAppName("startingSpark").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<Integer> myRDD = sc.parallelize(inputData);
        Integer result = myRDD.reduce(Integer::sum);

        myRDD.collect().forEach( System.out::println );
        myRDD.foreach(integer -> System.out.println(integer));
        System.out.println(result);
        /**
         * 25
         * 25
         * 25
         */

        sc.close();

    }
}
