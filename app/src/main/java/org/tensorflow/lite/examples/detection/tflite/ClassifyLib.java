package org.tensorflow.lite.examples.detection.tflite;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ClassifyLib {
    private int[] ddims = {1, 3, 224, 224};//根据自己的实际情况修改
    private List<String> resultLabel = new ArrayList<>();//用来存取标签值
    private Interpreter tflite = null;//新建一个全局 解译器对象 用来加载模型  运行模型  释放模型
    private String modelname = "fairface-int8";//修改为自己的模型名称

//-------------------------------------------加载分类模型-------------------------------------------------------------------
    //从assets文件中读取模型文件 谷歌官方提供的读取tflite文件的方法 与Interpreter配合 常规操作
    private MappedByteBuffer loadModelFile( Context context) throws IOException {
        //获取通过openFd()的方法获取asset目录下指定文件的AssetFileDescriptor对象
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelname + ".tflite");
        //返回可用于读取文件中的数据的FileDescriptor对象
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();//返回与此文件输入流关联的通道
        long startOffset = fileDescriptor.getStartOffset();//返回asset中项的数据开始处的字节偏移量
        long declaredLength = fileDescriptor.getDeclaredLength();//返回构造AssetFileDescriptor时声明的实际字节数
        //map方法来把文件影射为内存映像文件 把文件的从position开始的size大小的区域映射为内存映像文件，
        // mode指出了 可访问该内存映像文件的方式：READ_ONLY，READ_WRITE，PRIVATE
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);//
    }
    // load infer model
    public boolean load_model(Context context) {
        try {

            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4);//4线程运行
            tflite = new Interpreter(loadModelFile(context), options);
            return true;

        } catch (IOException e) {
            //e.printStackTrace();
            return false;
        }
    }
//-------------------------------------------加载分类模型-------------------------------------------------------------------


//-------------------------------------------加载分类标签-------------------------------------------------------------------
    // 从assets文件夹中读取标签文件 按行写入到resultLabel 列表
    public void readCacheLabelFromLocalFile(Context context) {
        try {
            AssetManager assetManager = context.getAssets();
            BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open("class.txt")));
            String readLine = null;
            while ((readLine = reader.readLine()) != null) {
                resultLabel.add(readLine);
            }
            reader.close();
        } catch (Exception e) {
            Log.e("labelCache", "error " + e);
        }
    }
//-------------------------------------------加载分类标签-------------------------------------------------------------------


//-------------------------------------------输入数据预处理---------------------------------------------------------------
    private static ByteBuffer getScaledMatrix(Bitmap bitmap, int[] ddims) {
        //在Java中当我们要对数据进行更底层的操作时，一般是操作数据的字节（byte）形式
        //每个像素点的三个分量都是一个int类型 java中int占4个字节 因此我们预先分配ddims[0] * ddims[1] * ddims[2] * ddims[3] * 4
        ByteBuffer imgData = ByteBuffer.allocateDirect(ddims[0] * ddims[1] * ddims[2] * ddims[3] * 4);
        imgData.order(ByteOrder.nativeOrder());//imgData的字节序按照当前机器使用的字节序
        // get image pixel
        int[] pixels = new int[ddims[2] * ddims[3]];
        //将原图片按照ddims[2], ddims[3] 进行缩放 filter决定是否平滑
        Bitmap bm = Bitmap.createScaledBitmap(bitmap, ddims[2], ddims[3], false);
        //将bm中的每个像素颜色转为int值存入pixels 每隔bm.getWidth()个像素换一行 每个值都是一个十进制 有时候还是负数
        bm.getPixels(pixels, 0, bm.getWidth(), 0, 0, ddims[2], ddims[3]);
        // val -15001066 tmp1 =27 tmp2=26 tmp3 = 22
        int pixel = 0;
        for (int i = 0; i < ddims[2]; ++i) {
            for (int j = 0; j < ddims[3]; ++j) {
                final int val = pixels[pixel++];
//                imgData.putFloat(((((val >> 16) & 0xFF) - 128f) / 128f));
//                imgData.putFloat(((((val >> 8) & 0xFF) - 128f) / 128f));
//                imgData.putFloat((((val & 0xFF) - 128f) / 128f));
                int tmp1 = ((val >> 16) & 0xFF);
                int tmp2 = ((val >> 8) & 0xFF);
                int tmp3 = ((val ) & 0xFF);
                //预处理非常重要
                imgData.putFloat(((((val >> 16) & 0xFF)/ 255.0f) - 0.485f ) / 0.229f);
                imgData.putFloat(((((val >> 8) & 0xFF)/ 255.0f) - 0.456f ) / 0.224f);
                imgData.putFloat((((val & 0xFF)/ 255.0f) - 0.406f ) / 0.225f);
            }
        }

        if (bm.isRecycled()) {
            bm.recycle();
        }
        return imgData;
    }
//-------------------------------------------输入数据预处理---------------------------------------------------------------

//--------------------------------------------推理预测------------------------------------------------------------------
    //  predict image
    public String predict_image(Bitmap bmp) {

        ByteBuffer inputData = getScaledMatrix(bmp, ddims);
        String show_text = "" ;
        try {

            //原始模型输出的结果为10类 1行10列 属于10类的概率值 因此new一个float类型的数组用来存放run之后的结果
            //float[][] labelProbArray = new float[1][10]; //为输出数据建立与模型输出结果对应的变量
            final float[][] labelProbArray = new float[1][18]; //为输出数据建立与模型输出结果对应的变量
            Object[] inputArray = {inputData};
            Map<Integer, Object> outputMap = new HashMap(){
                {put(0,labelProbArray);}

            };
            long start = System.currentTimeMillis();
            // get predict result
            //tflite.run(inputData, labelProbArray);//运行模型
            tflite.runForMultipleInputsOutputs(inputArray,outputMap);
            long end = System.currentTimeMillis();
            long time = end - start;
            float[] results = new float[((float[][])outputMap.get(0))[0].length];
            //float[] results = new float[labelProbArray[0].length];
            //为了方便计算 将第一行的10个概率值拷贝到一个一维数组results
           // System.arraycopy(labelProbArray[0], 0, results, 0, labelProbArray[0].length);
            System.arraycopy(((float[][])outputMap.get(0))[0], 0, results, 0, ((float[][])outputMap.get(0))[0].length);

            int[] index = get_max_result(results);//输出数据后处理

//            show_text = "race：" + resultLabel.get(index[0]) + "\ngender：" + resultLabel.get(index[1]) + "\nage：" + resultLabel.get(index[2]) + "\ntime：" + time + "ms";
            show_text = resultLabel.get(index[0]) + " " + resultLabel.get(index[1]) + " " + resultLabel.get(index[2]);


        } catch (Exception e) {
            e.printStackTrace();
        }
        return show_text;
    }

//--------------------------------------------推理预测------------------------------------------------------------------

//-------------------------------------------模型释放与关闭------------------------------------------------------
    public void close(){

        tflite.close();

    }
//-------------------------------------------模型释放与关闭------------------------------------------------------


//--------------------------------------------------输出数据后处理-----------------------------------------------
    // get max probability label
    private int[] get_max_result(float[] result) {
        int[] index={0,7,9};
        float probability_race = result[0];
        float probability_gender = result[7];
        float probability_age = result[9];

        //通过一个循环找到概率值最大的类别索引
        for (int i = 0; i < result.length; i++) {
            if(i < 7){
                if (probability_race < result[i]) {
                    probability_race = result[i];
                    index[0] = i;
                }
            }
            if(i >=7 && i < 9)
            {
                if (probability_gender < result[i]) {
                    probability_gender = result[i];
                    index[1] = i;
                }
            }
           if(i >=9){
                if (probability_age < result[i]) {
                    probability_age = result[i];
                    index[2] = i;
                }
            }
        }

        return index;
    }
//--------------------------------------------------输出数据后处理-----------------------------------------------


}
