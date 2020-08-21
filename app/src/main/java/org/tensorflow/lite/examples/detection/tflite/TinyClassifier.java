package org.tensorflow.lite.examples.detection.tflite;

import android.content.res.AssetManager;

import java.io.IOException;

/**
 * Created by Zuojianhao on 2020/8/11.
 */

public class TinyClassifier extends Classifier {

    protected float mObjThresh = 0.1f;

    public TinyClassifier(AssetManager assetManager) throws IOException{
//        super(assetManager, "yolov3-tiny-kitchen_six_int8.tflite", "kitchen_six.txt", 416);
//        mAnchors = new int[]{
//                16, 41,  31, 85,  57,102,  75,204, 170,292, 300,371
//        };

        super(assetManager, "yolov3_tiny_widerface_boxes_int8.tflite", "face.txt", 416);
        mAnchors = new int[]{
                8, 15,  14, 26,  24, 42,  43, 70,  81,124, 162,218
        };

        mMasks = new int[][]{{3,4,5},{0,1,2}};
        mOutWidth = new int[]{13,26};
        mObjThresh = 0.1f;
    }

    @Override
    protected float getObjThresh() {
        return mObjThresh;
    }
}
