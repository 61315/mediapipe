// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

constexpr char kInputStream[] 		= "input_video";
constexpr char kOutputVideo[] 		= "output_video";
constexpr char kOutputCorpusMask[] 	= "output_corpus_mask";
constexpr char kOutputFaceMask[] 	= "output_face_mask";
constexpr char kOutputSelfieMask[] 	= "output_selfie_mask";
constexpr char kWindowName[] 		= "Inpainting";

/*
# ImageFormat::SRGB(=CV_8UC3)
output_stream: "output_video"

# ImageFormat::SRGB(=CV_8UC3)
output_stream: "output_corpus_mask"

# ImageFormat::SRGB(=CV_8UC3)
output_stream: "output_face_mask"

# ImageFormat::VEC32F1(=CV_32FC1), values scaled 0 to 1.
output_stream: "output_selfie_mask"
*/

/*
Christmas, Christmas time is near
Time for toys and time for cheer
We've been good, but we can't last
Hurry Christmas, hurry fast
Want a plane that loops the loop
Me, I want a hula-hoop
We can hardly stand the wait
Please, Christmas, don't be late
*/

ABSL_FLAG(std::string, calculator_graph_config_file, "",
		  "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "",
		  "Full path of video to load. "
		  "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
		  "Full path of where to save result (.mp4 only). "
		  "If not provided, show result in a window.");

absl::Status RunMPPGraph()
{
	std::string calculator_graph_config_contents;
	MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
		absl::GetFlag(FLAGS_calculator_graph_config_file),
		&calculator_graph_config_contents));
	LOG(INFO) << "Get calculator graph config contents: "
			  << calculator_graph_config_contents;
	mediapipe::CalculatorGraphConfig config =
		mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
			calculator_graph_config_contents);

	LOG(INFO) << "Initialize the calculator graph.";
	mediapipe::CalculatorGraph graph;
	MP_RETURN_IF_ERROR(graph.Initialize(config));

	LOG(INFO) << "Initialize the camera or load the video.";
	cv::VideoCapture capture;
	const bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
	if (load_video)
	{
		capture.open(absl::GetFlag(FLAGS_input_video_path));
	}
	else
	{
		capture.open(0);
	}
	RET_CHECK(capture.isOpened());

	cv::VideoWriter writer;
	const bool save_video = !absl::GetFlag(FLAGS_output_video_path).empty();
	if (!save_video)
	{
		cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
		capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
		capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
		capture.set(cv::CAP_PROP_FPS, 30);
#endif
	}

	LOG(INFO) << "Start running the calculator graph.";

	// TODO: Use single poller that polls a single
	// output stream that contains multiple `ImageFrame`s.
	ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_video,
					 graph.AddOutputStreamPoller(kOutputVideo));
	ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_corpus_mask,
					 graph.AddOutputStreamPoller(kOutputCorpusMask));
	ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_face_mask,
					 graph.AddOutputStreamPoller(kOutputFaceMask));
	ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_selfie_mask,
					 graph.AddOutputStreamPoller(kOutputSelfieMask));
	MP_RETURN_IF_ERROR(graph.StartRun({}));

	LOG(INFO) << "Start grabbing and processing frames.";
	bool grab_frames = true;
	size_t frame_count = 0;
	while (grab_frames)
	{
		// Capture opencv camera or video frame.
		cv::Mat camera_frame_raw;
		capture >> camera_frame_raw;
		if (camera_frame_raw.empty())
		{
			if (!load_video)
			{
				LOG(INFO) << "Ignore empty frames from camera.";
				continue;
			}
			LOG(INFO) << "Empty frame, end of video reached.";
			break;
		}
		cv::Mat camera_frame;
		cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
		if (!load_video)
		{
			cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
		}

		// Wrap Mat into an ImageFrame.
		auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
			mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
			mediapipe::ImageFrame::kDefaultAlignmentBoundary);
		cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
		camera_frame.copyTo(input_frame_mat);

		// Send image packet into the graph.
		size_t frame_timestamp_us =
			(double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
		MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
			kInputStream, mediapipe::Adopt(input_frame.release())
							  .At(mediapipe::Timestamp(frame_timestamp_us))));

		// Get the graph result packet, or stop if that fails.
		mediapipe::Packet packet_video, packet_corpus_mask, packet_face_mask, packet_selfie_mask;
		if (!poller_video
				 .Next(&packet_video) ||
			!poller_corpus_mask
				 .Next(&packet_corpus_mask) ||
			!poller_face_mask
				 .Next(&packet_face_mask) ||
			!poller_selfie_mask
				 .Next(&packet_selfie_mask))
			break;
		auto &output_video = packet_video.Get<mediapipe::ImageFrame>();

		// Convert back to opencv for display or saving.
		cv::Mat output_video_mat = mediapipe::formats::MatView(&output_video);
		cv::cvtColor(output_video_mat, output_video_mat, cv::COLOR_RGB2BGR);

		// Get the corpus mask mat.
		auto &output_corpus_mask = packet_corpus_mask.Get<mediapipe::ImageFrame>();
		cv::Mat output_corpus_mask_mat = mediapipe::formats::MatView(&output_corpus_mask);

		// Prepare the face mask.
		auto &output_face_mask = packet_face_mask.Get<mediapipe::ImageFrame>();
		cv::Mat output_face_mask_mat = mediapipe::formats::MatView(&output_face_mask);
		cv::floodFill(output_face_mask_mat, cv::Point(0, 0), cv::Scalar(255, 255, 255));
		cv::bitwise_not(output_face_mask_mat, output_face_mask_mat);

		// Prepare the selfie mask.
		auto &output_selfie_mask = packet_selfie_mask.Get<mediapipe::ImageFrame>();
		cv::Mat output_selfie_mask_mat = mediapipe::formats::MatView(&output_selfie_mask);
		output_selfie_mask_mat.convertTo(output_selfie_mask_mat, CV_8U, 255);
		cv::cvtColor(output_selfie_mask_mat, output_selfie_mask_mat, CV_GRAY2RGB);
		cv::threshold(output_selfie_mask_mat, output_selfie_mask_mat, 192, 255, CV_THRESH_BINARY);

		// Do arithmetic operations to get the inpainting area mask.
		cv::Mat inpainting_mask;
		cv::bitwise_and(output_selfie_mask_mat, output_corpus_mask_mat, inpainting_mask);
		cv::subtract(inpainting_mask, output_face_mask_mat, inpainting_mask);

		// Overlay the inpainting mask to the original video feed.
		cv::add(output_video_mat, inpainting_mask, output_video_mat);

		if (save_video)
		{
			if (!writer.isOpened())
			{
				LOG(INFO) << "Prepare video writer.";
				writer.open(absl::GetFlag(FLAGS_output_video_path),
							mediapipe::fourcc('a', 'v', 'c', '1'), // .mp4
							capture.get(cv::CAP_PROP_FPS), output_video_mat.size());
				RET_CHECK(writer.isOpened());
			}

			// Write the first 100 frames to a video if flag `output_video_path` is set.
			writer.write(output_video_mat);
			LOG(INFO) << "Writing frame " << frame_count << "...";
			frame_count++;
			if (frame_count > 99)
			{
				grab_frames = false;
			}
		}
		else
		{
			cv::imshow(kWindowName, output_video_mat);
			// Press any key to exit.
			const int pressed_key = cv::waitKey(5);
			if (pressed_key >= 0 && pressed_key != 255)
				grab_frames = false;
		}
	}

	LOG(INFO) << "Shutting down.";
	if (writer.isOpened())
		writer.release();
	MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
	return graph.WaitUntilDone();
}

int main(int argc, char **argv)
{
	google::InitGoogleLogging(argv[0]);
	absl::ParseCommandLine(argc, argv);
	absl::Status run_status = RunMPPGraph();
	if (!run_status.ok())
	{
		LOG(ERROR) << "Failed to run the graph: " << run_status.message();
		return EXIT_FAILURE;
	}
	else
	{
		LOG(INFO) << "Success!";
	}
	return EXIT_SUCCESS;
}
