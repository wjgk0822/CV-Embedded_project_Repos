#include <iostream>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

using namespace std;
using namespace cv;
using namespace dnn;

void pytesseract_ocr(Mat& frame, int x1, int y1, int x2, int y2);

int main() {
    // YOLOv5 모델 경로 설정
    string yolo_model_path = "yolov5s.onnx";

    // OpenCV VideoCapture 초기화
    VideoCapture cap("2.mp4");

    if (!cap.isOpened()) {
        cerr << "Error opening video file." << endl;
        return -1;
    }

    // YOLOv5 모델 로드
    Net model = readNet(yolo_model_path);

    vector<string> class_name = { "Licence" };

    tesseract::TessBaseAPI tess;
    tess.Init(NULL, "eng");

    int frame_width = int(cap.get(3));
    int frame_height = int(cap.get(4));

    VideoWriter out("output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height));

    int count = 0;

    while (true) {
        Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

        count++;

        // YOLOv5를 사용하여 객체 탐지
        Mat blob = blobFromImage(frame, 1 / 255.0, Size(640, 640), Scalar(0, 0, 0), true, false);
        model.setInput(blob);
        Mat output = model.forward();

        // 탐지된 객체를 화면에 표시
        for (int i = 0; i < output.rows; ++i) {
            Point classIdPoint;
            double confidence;
            Rect box;

            auto data = (float*)output.row(i).data;
            classIdPoint.x = (int)data[0];
            confidence = data[1];
            box.x = (int)(data[2] * frame.cols);
            box.y = (int)(data[3] * frame.rows);
            box.width = (int)(data[4] * frame.cols);
            box.height = (int)(data[5] * frame.rows);

            // 신뢰도가 임계값을 초과하는 경우 객체로 간주
            if (confidence > 0.35) {
                int x1 = box.x;
                int y1 = box.y;
                int x2 = box.x + box.width;
                int y2 = box.y + box.height;

                cout << "Frame Count: " << count << " BBox: " << x1 << " " << y1 << " " << x2 << " " << y2 << endl;

                int classIndex = classIdPoint.x;
                string className = class_name[classIndex];
                string text;
                pytesseract_ocr(frame, x1, y1, x2, y2);

                rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(225, 10, 30), 3);
                Size textSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, 1, 2, nullptr);
                Point c2 = Point(x1 + textSize.width, y1 - textSize.height - 3);
                rectangle(frame, Point(x1, y1), c2, Scalar(225, 10, 30), -1, LINE_AA);
                putText(frame, text, Point(x1, y1 - 2), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, LINE_AA);
            }
        }

        out.write(frame);
        imshow("Frame", frame);

        if (waitKey(1) == 27)  // 'ESC' 키를 누르면 종료
            break;
    }

    cap.release();
    out.release();
    destroyAllWindows();

    return 0;
}

void pytesseract_ocr(Mat& frame, int x1, int y1, int x2, int y2) {
    Mat roi = frame(Rect(x1, y1, x2 - x1, y2 - y1));
    cvtColor(roi, roi, COLOR_BGR2RGB);

    tesseract::TessBaseAPI tess;
    tess.Init(NULL, "eng");
    tess.SetImage(roi.data, roi.cols, roi.rows, 3, roi.step);
    tess.Recognize(0);
    tesseract::ResultIterator* ri = tess.GetIterator();
    string result;

    do {
        const char* word = ri->GetUTF8Text(tesseract::RIL_WORD);
        result += word ? word : "";
        result += " ";
        delete[] word;
    } while (ri->Next(tesseract::RIL_WORD));

    std::replace(result.begin(), result.end(), 'O', '0');
    std::replace(result.begin(), result.end(), '_', ' ');
    std::replace(result.begin(), result.end(), '?', ' ');

    cout << "OCR Result: " << result << endl;
}
