/*
 * Ubitrack - Library for Ubiquitous Tracking
 * Copyright 2006, Technische Universitaet Muenchen, and individual
 * contributors as indicated by the @authors tag. See the
 * copyright.txt in the distribution for a full listing of individual
 * contributors.
 *
 * This is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of
 * the License, or (at your option) any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this software; if not, write to the Free
 * Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
 * 02110-1301 USA, or see the FSF site: http://www.fsf.org.
 */

/**
 * @ingroup vision_components
 * @file
 * Reads stereo camera images and intrinsics to generate a depth map
 *
 * @author Philipp Müller
 * @TODO:	- Delete bad image pairs in vector m_imageList too reduce computation time
 *			- Get 2D Points from Ubitrack Components
 *			- Performance: Use boost smart pointers instead of .Clone() for debug image outputs
 *			- DESYNCH ISSUES IF YOU START CALIBRATION IN FAST SUCCESSION, MAYBE MUTEX LOCK
 */

#include <string>
#include <list>
#include <iostream>
#include <iomanip>
#include <strstream>
#include <log4cpp/Category.hh>

#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/scoped_array.hpp>

#include <utDataflow/TriggerComponent.h>
#include <utDataflow/TriggerInPort.h>
#include <utDataflow/TriggerOutPort.h>
#include <utDataflow/ComponentFactory.h>
#include <utMeasurement/Measurement.h>


#include <utUtil/TracingProvider.h>
#include <opencv/cv.h>
#include <utVision/Image.h>

// OpenCV calib
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <algorithm>
#include <iterator>
#include <ctype.h>
// OpenCV calib end


// get a logger
static log4cpp::Category& logger(log4cpp::Category::getInstance("Ubitrack.Vision.StereoCameraCalibrationOpenCV"));

using namespace Ubitrack;
using namespace Ubitrack::Vision;

namespace Ubitrack { namespace Drivers {

/**
* @ingroup vision_components
*
* @par Input Ports
* None.
*
* @par Output Ports
* \c Output push port of type Ubitrack::Measurement::ImageMeasurement.
*
* @par Configuration
* The configuration tag contains a \c <dsvl_input> configuration.
* For details, see the DirectShow documentation...
*
*/
class StereoCameraCalibrationOpenCVComponent
	: public Dataflow::TriggerComponent
{
public:
	StereoCameraCalibrationOpenCVComponent(const std::string& sName, boost::shared_ptr< Graph::UTQLSubgraph >);
	~StereoCameraCalibrationOpenCVComponent();

	/** ubitrack start */
	void start();

	/** ubitrack stop */
	void stop();

	void compute(Measurement::Timestamp t);

	void stereoCalibration(Measurement::Timestamp timeStamp, bool useCalibrated = true);

protected:
	// Input edges
	Dataflow::TriggerInPort< Measurement::ImageMeasurement > m_grayImageLeft;
	Dataflow::TriggerInPort< Measurement::ImageMeasurement > m_grayImageRight;
	Dataflow::TriggerInPort< Measurement::PositionList > m_objectPoints;
	// Output edges
	//Dataflow::TriggerOutPort< Measurement::ImageMeasurement > m_outDebugImage;
	Dataflow::TriggerOutPort< Measurement::ImageMeasurement > m_outCornerImage;
	Dataflow::PushSupplier< Measurement::ImageMeasurement > m_outRectifiedImage;
	Dataflow::PushSupplier< Measurement::CameraIntrinsics > m_outIntrinsicsLeft;
	Dataflow::PushSupplier< Measurement::CameraIntrinsics > m_outIntrinsicsRight;
	//Dataflow::PushSupplier< Measurement::Vector4D > m_outDistortionLeft;
	//Dataflow::PushSupplier< Measurement::Vector4D > m_outDistortionRight;
	Dataflow::PushSupplier< Measurement::Pose > m_outRelativePose;

private:
	std::vector< Measurement::ImageMeasurement > m_imageList;
};


StereoCameraCalibrationOpenCVComponent::StereoCameraCalibrationOpenCVComponent(const std::string& sName, boost::shared_ptr< Graph::UTQLSubgraph > subgraph)
	: Dataflow::TriggerComponent(sName, subgraph)
	, m_grayImageLeft("GrayImageLeft", *this)
	, m_grayImageRight("GrayImageRight", *this)
	, m_objectPoints("GridPoints", *this)
	, m_outCornerImage("CornerImage", *this)
	, m_outRectifiedImage("RectifiedImage", *this)
	, m_outIntrinsicsLeft("IntrisicsLeft", *this)
	, m_outIntrinsicsRight("IntrisicsRight", *this)
	//, m_outDistortionLeft("DistortionLeft", *this)
	//, m_outDistortionRight("DistortionRight", *this)
	, m_outRelativePose("RelativePose", *this)
{

}

StereoCameraCalibrationOpenCVComponent::~StereoCameraCalibrationOpenCVComponent()
{

}

void StereoCameraCalibrationOpenCVComponent::start()
{
	Component::start();
}

void StereoCameraCalibrationOpenCVComponent::stop()
{
	Component::stop();
}

void StereoCameraCalibrationOpenCVComponent::compute(Measurement::Timestamp t)
{
	const Measurement::ImageMeasurement imageLeft = m_grayImageLeft.get();
	const Measurement::ImageMeasurement imageRight = m_grayImageRight.get();

	m_imageList.push_back(imageLeft);	// Left image
	m_imageList.push_back(imageRight);	// Right image

	LOG4CPP_INFO(logger, "Starting stereo calibration with " << m_imageList.size() / 2 << " image pairs.");

	stereoCalibration(true);
}

void StereoCameraCalibrationOpenCVComponent::stereoCalibration(Measurement::Timestamp timeStamp, bool useCalibrated)
{
	if (m_imageList.size() % 2 != 0)
	{
		LOG4CPP_ERROR(logger, "Error: the image list contains odd (non-even) number of elements.");
		return;
	}

	const int maxScale = 2;

	// DEBUG BOARD SIZE, REFACTOR 2D POINT DETECTION OUTSIDE OF STEREO CALIB
	cv::Size boardSize;
	boardSize.width = 9;
	boardSize.height = 6;
	// DEBUG END
	
	// ARRAY AND VECTOR STORAGE:
	std::vector<std::vector<cv::Point2f> > imagePoints[2];
	cv::Size imageSize;

	int i, k;
	int j = 0;	// number of good image pairs
	int nimages = (int)m_imageList.size() / 2;

	imagePoints[0].resize(nimages);	// left image points
	imagePoints[1].resize(nimages);	// right image points
	std::vector<Measurement::ImageMeasurement> goodImageList;

	for (i = 0; i < nimages; i++)
	{
		// Alternating left and right image
		for (k = 0; k < 2; k++)
		{
			//const std::string& filename = imagelist[i * 2 + k];
			//cv::Mat img = cv::imread(filename, 0);
			cv::Mat img = m_imageList[i * 2 + k].get()->Mat();
			if (img.empty())
				break;
			// For first non empty image save the image size
			if (imageSize == cv::Size())
				imageSize = img.size();
			else if (img.size() != imageSize)
			{
				LOG4CPP_ALERT(logger, "The image has a different size from the first image size. Skipping the pair");
				break;
			}
			// Find chessboard corners and save them in corners list
			bool found = false;
			std::vector<cv::Point2f>& corners = imagePoints[k][j];
			for (int scale = 1; scale <= maxScale; scale++)
			{
				cv::Mat timg;
				if (scale == 1)
					timg = img;
				else
					cv::resize(img, timg, cv::Size(), scale, scale);
				found = cv::findChessboardCorners(timg, boardSize, corners,
					cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
				if (found)
				{
					if (scale > 1)
					{
						cv::Mat cornersMat(corners);
						cornersMat *= 1. / scale;
					}
					break;
				}
			}
			// Display corners in a seperate window
			if (m_outCornerImage.isConnected() && i == nimages - 1 && k == 0)
			{
				cv::Mat cimg, cimg1;
				cv::cvtColor(img, cimg, cv::COLOR_GRAY2BGR);
				cv::drawChessboardCorners(cimg, boardSize, corners, found);
				double sf = 640.0 / MAX(img.rows, img.cols);
				cv::resize(cimg, cimg1, cv::Size(), sf, sf);
				m_outCornerImage.send(Measurement::ImageMeasurement(timeStamp, ImagePtr(new Image(cimg1))));
			}

			if (!found)
				break;
			// If we found corners we refine them to sub pixel accurate points
			cv::cornerSubPix(img, corners, cv::Size(11, 11), cv::Size(-1, -1),
				cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
				30, 0.01));
		}
		// If no break case was taken k == 2
		if (k == 2)
		{
			goodImageList.push_back(m_imageList[i * 2]);
			goodImageList.push_back(m_imageList[i * 2 + 1]);
			j++;
		}
	}

	LOG4CPP_INFO(logger, j << " pairs have been successfully detected.");
	// Set nimages to the number of good image pairs
	nimages = j;
	if (nimages < 2)
	{
		LOG4CPP_ERROR(logger, "Too little pairs to run the calibration.");
		return;
	}

	// Resize left and right imagePoints to number of good image pairs found before
	// imagePoints contains corner data
	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);

	std::vector< std::vector<cv::Point3f> > objectPoints;	// Chessboard points
	objectPoints.resize(nimages);	// Here first time used
	
	const std::vector< Math::Vector<double, 3> >& points3D = *m_objectPoints.get();
	// For each image pair
	for (i = 0; i < nimages; i++)
	{
		objectPoints[i].resize(points3D.size());
		for (j = 0; j < points3D.size(); j++)
		{
			objectPoints[i][j] = cv::Point3f(static_cast<float>(points3D[j][0]),
				static_cast<float>(points3D[j][1]), static_cast<float>(points3D[j][2])/*0.0f*/);
		}
	}

	LOG4CPP_INFO(logger, "Running stereo calibration ...");
	// cameraMatrix: intrinsic camera data
	// distCoeffs: distortion coefficients
	cv::Mat cameraMatrix[2], distCoeffs[2];
	// Init both camera matrices based on computed image points
	cameraMatrix[0] = cv::initCameraMatrix2D(objectPoints, imagePoints[0], imageSize, 0);
	cameraMatrix[1] = cv::initCameraMatrix2D(objectPoints, imagePoints[1], imageSize, 0);
	// R: Rotation matrix
	// T: Translation matrix
	// E: Essential matrix
	// F: Fundamental matrix
	cv::Mat R, T, E, F;

	// RMS: Root Mean Square error https://en.wikipedia.org/wiki/Root_mean_square
	double rms = cv::stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
		cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, E, F,
		cv::CALIB_FIX_ASPECT_RATIO +
		cv::CALIB_ZERO_TANGENT_DIST +
		cv::CALIB_USE_INTRINSIC_GUESS +
		cv::CALIB_SAME_FOCAL_LENGTH +
		cv::CALIB_RATIONAL_MODEL +
		cv::CALIB_FIX_K3 + cv::CALIB_FIX_K4 + cv::CALIB_FIX_K5,		// Distortion model approximation used
		cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));
	LOG4CPP_INFO(logger, "RMS error= " << rms);

	// CALIBRATION QUALITY CHECK
	// Because the output fundamental matrix implicitly
	// includes all the output information,
	// we can check the quality of calibration using the
	// epipolar geometry constraint: m2^t*F*m1=0
	double err = 0;
	int npoints = 0;
	std::vector<cv::Vec3f> lines[2];
	for (i = 0; i < nimages; i++)
	{
		int npt = (int)imagePoints[0][i].size();
		cv::Mat imgpt[2];
		for (k = 0; k < 2; k++)
		{
			imgpt[k] = cv::Mat(imagePoints[k][i]);
			cv::undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], cv::Mat(), cameraMatrix[k]);
			cv::computeCorrespondEpilines(imgpt[k], k + 1, F, lines[k]);
		}
		for (j = 0; j < npt; j++)
		{
			double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
				imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
				fabs(imagePoints[1][i][j].x*lines[0][j][0] +
				imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
			err += errij;
		}
		npoints += npt;
	}
	LOG4CPP_INFO(logger, "Average epipolar error = " << err / npoints);

	// Save intrinsic parameters
	m_outIntrinsicsLeft.send(Measurement::CameraIntrinsics(timeStamp, Math::CameraIntrinsics<double>(Math::Matrix<double, 3, 3>((double*)cameraMatrix[0].data),
		Math::Vector4d((double*)distCoeffs[0].data), m_imageList[0].get()->width(), m_imageList[0].get()->height())));
	m_outIntrinsicsRight.send(Measurement::CameraIntrinsics(timeStamp, Math::CameraIntrinsics<double>(Math::Matrix<double, 3, 3>((double*)cameraMatrix[1].data),
		Math::Vector4d((double*)distCoeffs[1].data), m_imageList[1].get()->width(), m_imageList[1].get()->height())));

	//cv::FileStorage fs("C:\\Users\\FAR-Student\\Desktop\\Phil\\CalibDataset\\intrinsics.yml", cv::FileStorage::WRITE);
	//if (fs.isOpened())
	//{
	//	fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
	//		"M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
	//	fs.release();
	//}
	//else
	//	LOG4CPP_ERROR(logger, "Error: can not save the intrinsic parameters.");

	cv::Mat R1, R2, P1, P2, Q;
	cv::Rect validRoi[2];

	cv::stereoRectify(cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, R1, R2, P1, P2, Q,
		cv::CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

	// Save extrinsic parameters
	Math::Pose poseCam1ToCam2 = Math::Pose(Math::Quaternion(Math::Matrix<double, 3, 3>((double*)R.data)), Math::Vector3d((double*)T.data));
	m_outRelativePose.send(Measurement::Pose(timeStamp, poseCam1ToCam2));

	//fs.open("C:\\Users\\FAR-Student\\Desktop\\Phil\\CalibDataset\\extrinsics.yml", cv::FileStorage::WRITE);
	//if (fs.isOpened())
	//{
	//	fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
	//	fs.release();
	//}
	//else
	//	LOG4CPP_ERROR(logger, "Error: can not save the extrinsic parameters.");

	// OpenCV can handle left-right
	// or up-down camera arrangements
	bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

	// COMPUTE AND DISPLAY RECTIFICATION
	if (!m_outRectifiedImage.isConnected())
		return;

	cv::Mat rmap[2][2];
	// IF BY CALIBRATED (BOUGUET'S METHOD)
	if (useCalibrated)
	{
		// we already computed everything
	}
	// OR ELSE HARTLEY'S METHOD
	else
	// Use intrinsic parameters of each camera, but
	// compute the rectification transformation directly
	// from the fundamental matrix
	{
		std::vector<cv::Point2f> allimgpt[2];
		for (k = 0; k < 2; k++)
		{
			for (i = 0; i < nimages; i++)
				std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
		}
		// Find fundamental matrix F
		F = cv::findFundamentalMat(cv::Mat(allimgpt[0]), cv::Mat(allimgpt[1]), cv::FM_8POINT, 0, 0);
		cv::Mat H1, H2;
		stereoRectifyUncalibrated(cv::Mat(allimgpt[0]), cv::Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

		R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
		R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
		P1 = cameraMatrix[0];
		P2 = cameraMatrix[1];
	}

	// Precompute maps for cv::remap()
	cv::initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	cv::initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	cv::Mat canvas;
	double sf;
	int w, h;
	if (!isVerticalStereo)
	{
		sf = 600. / MAX(imageSize.width, imageSize.height);
		w = cvRound(imageSize.width*sf);
		h = cvRound(imageSize.height*sf);
		canvas.create(h, w * 2, CV_8UC3);
	}
	else
	{
		sf = 300. / MAX(imageSize.width, imageSize.height);
		w = cvRound(imageSize.width*sf);
		h = cvRound(imageSize.height*sf);
		canvas.create(h * 2, w, CV_8UC3);
	}

	//for (i = 0; i < nimages; i++)
	for (i = nimages-1; i < nimages; i++)
	{
		for (k = 0; k < 2; k++)
		{
			cv::Mat img = goodImageList[i * 2 + k].get()->Mat(), rimg, cimg;
			cv::remap(img, rimg, rmap[k][0], rmap[k][1], cv::INTER_LINEAR);
			cv::cvtColor(rimg, cimg, cv::COLOR_GRAY2BGR);
			cv::Mat canvasPart = !isVerticalStereo ? canvas(cv::Rect(w*k, 0, w, h)) : canvas(cv::Rect(0, h*k, w, h));
			resize(cimg, canvasPart, canvasPart.size(), 0, 0, cv::INTER_AREA);
			if (useCalibrated)
			{
				cv::Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
					cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
				rectangle(canvasPart, vroi, cv::Scalar(0, 0, 255), 3, 8);
			}
		}

		if (!isVerticalStereo)
			for (j = 0; j < canvas.rows; j += 16)
				line(canvas, cv::Point(0, j), cv::Point(canvas.cols, j), cv::Scalar(0, 255, 0), 1, 8);
		else
			for (j = 0; j < canvas.cols; j += 16)
				line(canvas, cv::Point(j, 0), cv::Point(j, canvas.rows), cv::Scalar(0, 255, 0), 1, 8);

		m_outRectifiedImage.send(Measurement::ImageMeasurement(timeStamp, ImagePtr(new Image(canvas))));
	}
}

} } // namespace Ubitrack::Driver

UBITRACK_REGISTER_COMPONENT(Dataflow::ComponentFactory* const cf) {
	cf->registerComponent< Ubitrack::Drivers::StereoCameraCalibrationOpenCVComponent >("StereoCameraCalibrationOpenCVComponent");
}

