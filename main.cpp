#include <opencv2/opencv.hpp>
#include <iostream>

int main() {

    cv::VideoCapture camera(0);

    if (!camera.isOpened()) {
        std::cout << "Could not open camera." << std::endl;
        return 1;
    }

    cv::Mat frame;

    auto detector = cv::FaceDetectorYN::create(
        "face_detection_yunet_2023mar.onnx",
        "",
        cv::Size(320, 320)
    );

    auto recognizer = cv::FaceRecognizerSF::create(
        "face_recognition_sface_2021dec.onnx",
        ""
    );

    cv::Mat savedEmbedding;
    bool faceSaved = false;

    while (true) {

        camera >> frame;

        if (frame.empty()) {
            std::cout << "Could not read frame." << std::endl;
            break;
        }

        cv::flip(frame, frame, 1);

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        cv::Mat faces;

        detector->setInputSize(frame.size());
        detector->detect(frame, faces);

        std::string faceCount =
            "Faces found: " + std::to_string(faces.rows);

        cv::putText(
            frame,
            faceCount,
            cv::Point(20, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(255, 0, 0),
            2
        );

        cv::Mat currentEmbedding;

        for (int i = 0; i < faces.rows; i++) {

            int x = static_cast<int>(faces.at<float>(i, 0));
            int y = static_cast<int>(faces.at<float>(i, 1));
            int width = static_cast<int>(faces.at<float>(i, 2));
            int height = static_cast<int>(faces.at<float>(i, 3));

            cv::Rect faceBox(x, y, width, height);

            cv::rectangle(
                frame,
                faceBox,
                cv::Scalar(255, 0, 0),
                2
            );

            cv::Mat alignedFace;

            recognizer->alignCrop(
                frame,
                faces.row(i),
                alignedFace
            );

            cv::Mat embedding;

            recognizer->feature(
                alignedFace,
                embedding
            );

            currentEmbedding = embedding.clone();

            if (faceSaved) {

                double score = recognizer->match(
                    savedEmbedding,
                    embedding,
                    cv::FaceRecognizerSF::FR_COSINE
                );

                double threshold = 0.5;

                std::string identity;
                cv::Scalar textColor;

                if (score > threshold) {
                    identity = "Dalia";
                    textColor = cv::Scalar(0, 255, 0);
                } else {
                    identity = "Not Dalia";
                    textColor = cv::Scalar(0, 0, 255);
                }

                cv::putText(
                    frame,
                    identity,
                    cv::Point(x, y - 10),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    textColor,
                    2
                );

                std::cout
                    << "Similarity score: "
                    << score
                    << std::endl;
            }
        }

        cv::imshow("Camera", frame);

        int key = cv::waitKey(1);

        if (
            (key == 's' || key == 'S')
            && !currentEmbedding.empty()
        ) {
            savedEmbedding = currentEmbedding.clone();
            faceSaved = true;

            std::cout
                << "Face saved successfully!"
                << std::endl;
        }

        if (key == 27) {
            break;
        }
    }

    camera.release();
    cv::destroyAllWindows();

    return 0;
}