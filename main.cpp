#include <NetworkManager.h>
#include <csignal>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/opencv.hpp>
#include<stdlib.h>
#include<math.h>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
using namespace ml;
ostringstream oss;
bool reg = 0;
//127.0.0.1
//192.168.1.3
void error_handle(int error_id, std::string message);

Net::NetworkManager net("192.168.1.3", 20214538,"斯国一", 25562, 25564, error_handle);

void sigint_handler(int sig) { exit(1); }


//由于在识别中的核心物体以及相关的物理特性是灯条，所以建一个灯条类
class LightDescriptor
{	    //在识别以及匹配到灯条的功能中需要用到旋转矩形的长宽偏转角面积中心点坐标等
public:float width, length, angle, area;
      cv::Point2f center;
public:
    LightDescriptor() {};
    //让得到的灯条套上一个旋转矩形，以方便之后对角度这个特殊因素作为匹配标准
    LightDescriptor(const cv::RotatedRect& light)
    {
        width = light.size.width;
        length = light.size.height;
        center = light.center;
        angle = light.angle;
        area = light.size.area();
    }
};


class LeftDescriptor
{	    //在识别以及匹配到灯条的功能中需要用到旋转矩形的长宽偏转角面积中心点坐标等
public:float width, length, angle, area;
      cv::Point2f center;
public:
    LeftDescriptor() {};
    //让得到的灯条套上一个旋转矩形，以方便之后对角度这个特殊因素作为匹配标准
    LeftDescriptor(const cv::RotatedRect& light)
    {
        width = light.size.width;
        length = light.size.height;
        center = light.center;
        angle = light.angle;
        area = light.size.area();
    }
};






int main() {
    signal(SIGINT, sigint_handler);
    reg = net.registerUser(cv::getTickCount());
    while (!reg) {
        std::cout << "Register failed, retrying..." << std::endl;
        reg = net.registerUser(0);
    }
    std::cout << "Register success" << std::endl;
    float yaw =0;
    float pitch =0;
    int order=0;

    float dpitch[1000];
    dpitch[0]=-3;
    int ipitch=0;

    cv::Mat img,img1;
    cv::Mat binary, channels[3],Gaussian,dilatee,roi;
    cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11), cv::Point(-1, -1));
    Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7), cv::Point(-1, -1));
    Mat element2 = getStructuringElement(cv::MORPH_RECT, cv::Size(25, 25));//设置内核2
    cv::Mat mask,imgHSV;
    cv::Scalar lower(113,0,214);
    cv::Scalar upper(180,255,255);
    cv::Mat img_clone;
    Ptr<SVM>SVM_params;
    bool enemy_color=true;
    SVM_params = SVM::load("/home/txf/number/numsvm.xml");




    int stateNum = 4;
    int measureNum = 2;
    KalmanFilter KF(stateNum, measureNum, 0);
    //Mat processNoise(stateNum, 1, CV_32F);
    Mat measurement = Mat::zeros(measureNum, 1, CV_32F);
    KF.transitionMatrix = (Mat_<float>(stateNum, stateNum) << 1, 0, 1, 0,//A 状态转移矩阵
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1);
    //这里没有设置控制矩阵B，默认为零
    setIdentity(KF.measurementMatrix);//H=[1,0,0,0;0,1,0,0] 测量矩阵
    setIdentity(KF.processNoiseCov, Scalar::all(1e-5));//Q高斯白噪声，单位阵
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));//R高斯白噪声，单位阵
    setIdentity(KF.errorCovPost, Scalar::all(1));//P后验误差估计协方差矩阵，初始化为单位阵
    randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));//初始化状态为随机值




   while (true) {

        

        while (!reg) {
            std::cout << "Register failed, retrying..." << std::endl;
            reg = net.registerUser(cv::getTickCount());
        }

        img = net.getNewestRecvMessage().img;
        

        if (!img.empty()) {
        
        
        
        cv::RotatedRect box;
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        vector<vector<Point>> conPoly(contours.size());
        Rect boundRect;
        vector<Point2f> boxPts(4);
        cv::Point2f vertex[4];
        cv::Point2f leftPoint[4];

        img1=img.clone();
        cout<<net.getNewestRecvMessage().bullet_speed<<endl;
        //cout<<net.getNewestRecvMessage().buff_over_time<<endl;
        if(net.getNewestRecvMessage().rest_time>100){
             
        yaw =1;
        pitch =dpitch[ipitch];
        ipitch++;
        if(net.getNewestRecvMessage().rest_time<=118&&net.getNewestRecvMessage().buff_over_time==0){
            dpitch[ipitch]=dpitch[ipitch-1]-0.2;
        }
        net.sendControlMessage(Net::SendStruct(yaw, pitch, 0, 20.0, 0, 0.0, 0.0, -1, -1));
        
        

        
        cv::inRange(img1,lower,upper,mask);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, element2);//闭运算
        cv::floodFill(mask, cv::Point(0, 0), cv::Scalar(0));//漫水法
        cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);//轮廓检测
        net.sendControlMessage(Net::SendStruct(yaw, pitch, 0, 20.0, 0, 0.0, 0.0, -1, -1));
        int dfshoot=0;
        for (int i = 0; i < contours.size(); i++) {
            // 求轮廓面积

            double area = cv::contourArea(contours[i]);


            cv::RotatedRect minRect = cv::minAreaRect(cv::Mat(contours[i]));
		    minRect.points(vertex);

            boundRect = boundingRect(Mat(contours[i]));
			

            // 去除较小轮廓&fitEllipse的限制条件
            if (area < 200 ||contours[i].size() <= 2||area > 50000)
                continue;//相当于就是把这段轮廓去除掉
            if (minRect.size.width / minRect.size.height < 1.3&&minRect.size.width / minRect.size.height>0.7||minRect.angle<5&&minRect.angle>-5)
                continue;
            
            rectangle(img1, boundRect.tl(), boundRect.br(), Scalar(0, 255, 0), 5);
            // for (int l = 0; l < 4; l++)
            // {
            //     cv::line(img1, vertex[l], vertex[(l + 1) % 4], cv::Scalar(255, 0, 0), 2);
            // }
            circle(img1, Point(265,305), 3, Scalar(34, 255, 255), -1);
            //rectangle(img1, Point(255,298), Point(265,308), Scalar(0, 255, 0), 5);
            if(boundRect.tl().x<260&&boundRect.br().x>260&&boundRect.tl().y<305&&boundRect.br().y>305){
                dfshoot++;
                
                net.sendControlMessage(Net::SendStruct(yaw, pitch, 1, 20.0, 0, 0.0, 0.0, -1, -1));
                waitKey(5);
            }
           } 
        }        
        if(net.getNewestRecvMessage().rest_time<=100){
        net.sendControlMessage(Net::SendStruct(yaw,pitch,0,-1,0,0,0,-1,-1));
        if(yaw<60||yaw>300){
        yaw=60;
        net.sendControlMessage(Net::SendStruct(yaw,pitch,0,-1,0,0,0,-1,-1));
        }
        cv::split(img, channels); //通道分离
        if(enemy_color==true){
            subtract(channels[2],channels[1],img);
        }else if(enemy_color==false){
            subtract(channels[0],channels[2],img);
        }
        if(enemy_color==true){
            cv::threshold(img,binary,100,255,cv::THRESH_BINARY);
        }
        
        //cv::threshold(img1, binary, 0, 255, cv::THRESH_OTSU); //大津法是自动分类 需要先取低阈值
        cv::GaussianBlur(binary, Gaussian, cv::Size(5, 5), 0);//滤波
        erode(Gaussian, Gaussian, element, Point(-1, -1), 1, 0);
        dilate(Gaussian, dilatee, element);
        cv::findContours(dilatee, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);//轮廓检测
        vector<LightDescriptor> lightInfos;//创建一个灯条类的动态数组
        vector<LeftDescriptor> leftInfos;
        

        if(contours.size()>=4){
        for (int i = 0; i < contours.size(); i++) {
            // 求轮廓面积

            double area = cv::contourArea(contours[i]);


            cv::RotatedRect minRect = cv::minAreaRect(cv::Mat(contours[i]));
		    minRect.points(vertex);

            boundRect = boundingRect(Mat(contours[i]));
			

            // 去除较小轮廓&fitEllipse的限制条件
            if (area < 500 ||contours[i].size() <= 2||area > 50000)
                continue;//相当于就是把这段轮廓去除掉
            if (minRect.size.width / minRect.size.height > 1.3||minRect.size.width / minRect.size.height<0.7)
                continue;
            // for (int l = 0; l < 4; l++)
            // {
            //     cv::line(img1, vertex[l], vertex[(l + 1) % 4], cv::Scalar(255, 0, 0), 2);
            // }
            lightInfos.push_back(LightDescriptor(minRect));


            
            
        }
        for (size_t i = 0; i < lightInfos.size(); i++) {
            for (size_t j = i + 1; (j < lightInfos.size()); j++) {
                LightDescriptor& leftLight = lightInfos[i];
                LightDescriptor& rightLight = lightInfos[j];
                float angleGap_ = abs(leftLight.angle - rightLight.angle);
                //由于灯条长度会因为远近而受到影响，所以按照比值去匹配灯条
                float LenGap_ratio = abs(leftLight.length - rightLight.length) / max(leftLight.length, rightLight.length);
                //左右灯条相距距离
                float dis = pow(pow((leftLight.center.x - rightLight.center.x), 2) + pow((leftLight.center.y - rightLight.center.y), 2), 0.5);
                //左右灯条长度的平均值
                float meanLen = (leftLight.length + rightLight.length) / 2;
                //左右灯条长度差比值
                float lengap_ratio = abs(leftLight.length - rightLight.length) / meanLen;
                //左右灯条中心点y的差值
                float yGap = abs(leftLight.center.y - rightLight.center.y);
                //y差比率
                float yGap_ratio = yGap / meanLen;
                //左右灯条中心点x的差值
                float xGap = abs(leftLight.center.x - rightLight.center.x);
                //x差比率
                float xGap_ratio = xGap / meanLen;
                //相距距离与灯条长度比值
                float ratio = dis / meanLen;
                //匹配不通过的条件
                if (angleGap_ > 5 ||
                    LenGap_ratio > 0.5 ||
                    lengap_ratio > 0.4 ||
                    yGap_ratio > 3.3 ||
                    yGap_ratio < 2.0 ||
                    xGap_ratio > 3.3 ||
                    xGap_ratio < 2.0 ||
                    ratio > 4.3 ||
                    ratio < 3.0) {
                    continue;
                }
                Point center = Point((leftLight.center.x + rightLight.center.x) / 2, (leftLight.center.y + rightLight.center.y) / 2);
                circle(img1, center, 7.5, Scalar(0, 0, 255), 5);


                Mat prediction = KF.predict();
                Point predict_pt = Point((int)prediction.at<float>(0), (int)prediction.at<float>(1));
                measurement.at<float>(0) = (float)center.x;
                measurement.at<float>(1) = (float)center.y;
                KF.correct(measurement);
                circle(img1, predict_pt, 3, Scalar(34, 255, 255), -1);
                //center.x = (int)prediction.at<float>(0);
                //center.y = (int)prediction.at<float>(1);


                Point windowcenter = Point(320,240);
                
                
                if(center.x>0&&center.x<640)
                {
                    if(center.x>windowcenter.x) yaw+=1.6;
                    if(center.y>windowcenter.y) pitch+=0.4;
                    if(center.x<windowcenter.x) yaw-=1.6;
                    if(center.y<windowcenter.y) pitch-=0.4;

                    net.sendControlMessage(Net::SendStruct(yaw, pitch, 0, 20.0, 0, 0.0, 0.0, -1, -1));
                    waitKey(10);
                }
                if(abs(center.x-windowcenter.x)<leftLight.width&&abs(center.y-windowcenter.y)<1.5*leftLight.width){
                    pitch-=6;
                    net.sendControlMessage(Net::SendStruct(yaw, pitch, 0, 20.0, 0, 0.0, 0.0, -1, -1));
                    if(yaw<60||yaw>300){
                       yaw=60;
                    }
                    net.sendControlMessage(Net::SendStruct(yaw, pitch, 1, 20.0, 0, 0.0, 0.0, -1, -1));
                    waitKey(20);
                    pitch+=6;
                    net.sendControlMessage(Net::SendStruct(yaw, pitch, 0, 20.0, 0, 0.0, 0.0, -1, -1));
                    waitKey(20);
                }
                else{      
                    if(yaw<60||yaw>300){
                        yaw=60;
                    }
                    if(pitch<-20){
                        pitch=-5;
                    }
                    if(pitch>20){
                        pitch=-5;
                    }
                        net.sendControlMessage(Net::SendStruct(yaw, pitch, 0, 20.0, 0, 0.0, 0.0, -1, -1));
                    
                }
                

               

                RotatedRect leftCard = RotatedRect(Point2f(center.x, center.y), 
                Size2f(leftLight.width*3.9, leftLight.length*3.9), leftLight.angle);
                leftCard.points(vertex);
                for (int l = 0; l < 4; l++)
                {
                    cv::line(img1, vertex[l], vertex[(l + 1) % 4], cv::Scalar(255, 0, 0), 2);
                }
                leftInfos.push_back(LeftDescriptor(leftCard));
                }
        }
        }
        if(contours.size()<4){
            yaw+=0.3;
            net.sendControlMessage(Net::SendStruct(yaw, pitch, 0, 20.0, 0, 0.0, 0.0, -1, -1));
        }
       
        }
        //cv::imshow("img1",dilatee);
        cv::imshow("img", img1);
        cv::waitKey(1);
        } else {
            std::cout << "Get an empty image" << std::endl;
            cv::waitKey(100);
        }
        
        /*
        Code Here! Just define yaw ,pitch ,beat ... which in SendStruct's elements  and will be sent to us. 
        */
        //net.sendControlMessage(Net::SendStruct(yaw, pitch, 1, 20.0, 0, 0.0, 0.0, -1, -1));
    }
    return 0;
}

void error_handle(int error_id, std::string message) {
    if (error_id == 5) {
        reg = net.registerUser(0);
    }
    std::cout << "Error: " << error_id << " " << message << std::endl;
 }
