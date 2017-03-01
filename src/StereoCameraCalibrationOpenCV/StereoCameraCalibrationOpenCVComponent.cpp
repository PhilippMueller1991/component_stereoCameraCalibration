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
 * @TODO:	
 *		- Get 2D Points from Ubitrack Components
 *		- Performance: Delete bad image pairs in vector m_imageList too reduce computation time
 *		- Performance: Use boost smart pointers instead of .Clone() for debug image outputs
 *		- desynch issues if you start calibration in fast succession, maybe mutex lock
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

	void stereoCalibration(Measurement::Timestamp timeStamp);

protected:
	// Input edges
	Dataflow::TriggerInPort< Measurement::PositionList > m_objectPoints;
	Dataflow::TriggerInPort< Measurement::PositionList2 > m_cornersLeft;
	Dataflow::TriggerInPort< Measurement::PositionList2 > m_cornersRight;
	// Output edges
	Dataflow::TriggerOutPort< Measurement::Pose > m_outRelativePose;
	Dataflow::PushSupplier< Measurement::CameraIntrinsics > m_outIntrinsicsLeft;
	Dataflow::PushSupplier< Measurement::CameraIntrinsics > m_outIntrinsicsRight;

private:
	//std::vector< Measurement::ImageMeasurement > m_imageList;
	std::vector<std::vector<cv::Point2f> > m_imagePoints[2];
};


StereoCameraCalibrationOpenCVComponent::StereoCameraCalibrationOpenCVComponent(const std::string& sName, boost::shared_ptr< Graph::UTQLSubgraph > subgraph)
	: Dataflow::TriggerComponent(sName, subgraph)
	, m_objectPoints("GridPoints", *this)
	, m_cornersLeft("CornersLeft", *this)
	, m_cornersRight("CornersRight", *this)
	, m_outIntrinsicsLeft("IntrisicsLeft", *this)
	, m_outIntrinsicsRight("IntrisicsRight", *this)
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
	if (m_cornersLeft.get()->size() != m_cornersRight.get()->size())
	{
		LOG4CPP_INFO(logger, "Number of found corners differs.");
		return;
	}

	// Left image corners
	m_imagePoints[0].resize(m_imagePoints[0].size() + 1);
	m_imagePoints[0][m_imagePoints[0].size() - 1].resize(m_cornersLeft.get()->size());
	for (int i = 0; i < m_cornersLeft.get()->size(); i++)
	{
		Ubitrack::Math::Vector2d& utCornerLeft = m_cornersLeft.get()->data()[i];
		m_imagePoints[0][m_imagePoints[0].size()-1][i] = cv::Point2f((float)utCornerLeft[0], (float)utCornerLeft[1]);
	}

	// Right image corners
	m_imagePoints[1].resize(m_imagePoints[1].size() + 1);
	m_imagePoints[1][m_imagePoints[1].size() - 1].resize(m_cornersRight.get()->size());
	for (int i = 0; i < m_cornersRight.get()->size(); i++)
	{
		Ubitrack::Math::Vector2d& utCornerRight = m_cornersRight.get()->data()[i];
		m_imagePoints[1][m_imagePoints[1].size() - 1][i] = cv::Point2f((float)utCornerRight[0], (float)utCornerRight[1]);
	}

	LOG4CPP_INFO(logger, "Starting stereo calibration with " << m_imagePoints[0].size() << " image pairs.");

	stereoCalibration(t);
}

void StereoCameraCalibrationOpenCVComponent::stereoCalibration(Measurement::Timestamp timeStamp)
{
	// DEBUG FOR NOW USE FIXED SIZE, LATER USE ATTRIBUTES
	const cv::Size imageSize = cv::Size(640, 480);
	int nimages = m_imagePoints[0].size();
	if (nimages < 2)
	{
		LOG4CPP_INFO(logger, "Too few pairs to run the calibration");
		return;
	}

	// Chessboard 3D points, where z is always 0
	std::vector< std::vector<cv::Point3f> > objectPoints;
	objectPoints.resize(nimages);
	
	const std::vector< Math::Vector<double, 3> >& points3D = *m_objectPoints.get();
	// For each image pair
	for (int i = 0; i < nimages; i++)
	{
		objectPoints[i].resize(points3D.size());
		for (int j = 0; j < points3D.size(); j++)
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
	cameraMatrix[0] = cv::initCameraMatrix2D(objectPoints, m_imagePoints[0], imageSize, 0);
	cameraMatrix[1] = cv::initCameraMatrix2D(objectPoints, m_imagePoints[1], imageSize, 0);
	cv::Mat R, T, E, F;

	// RMS: Root Mean Square error https://en.wikipedia.org/wiki/Root_mean_square
	double rms = cv::stereoCalibrate(objectPoints, m_imagePoints[0], m_imagePoints[1],
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
	for (int i = 0; i < nimages; i++)
	{
		int npt = (int)m_imagePoints[0][i].size();
		cv::Mat imgpt[2];
		for (int k = 0; k < 2; k++)
		{
			imgpt[k] = cv::Mat(m_imagePoints[k][i]);
			cv::undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], cv::Mat(), cameraMatrix[k]);
			cv::computeCorrespondEpilines(imgpt[k], k + 1, F, lines[k]);
		}
		for (int j = 0; j < npt; j++)
		{
			double errij = fabs(m_imagePoints[0][i][j].x*lines[1][j][0] +
				m_imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
				fabs(m_imagePoints[1][i][j].x*lines[0][j][0] +
				m_imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
			err += errij;
		}
		npoints += npt;
	}
	LOG4CPP_INFO(logger, "Average epipolar error = " << err / npoints);

	// Save intrinsic parameters
	Math::CameraIntrinsics<double> camIntrisics[2];
	for (int i = 0; i < 2; i++)
	{
		Math::Matrix<double, 3, 3> tmpIntrinsics = Math::Matrix<double, 3, 3>((double*)cameraMatrix[i].data);
		Math::Vector4d tmpDist = Math::Vector4d((double*)distCoeffs[i].data);
		camIntrisics[i] = Math::CameraIntrinsics<double>(tmpIntrinsics, tmpDist, imageSize.width, imageSize.height);
	}
	m_outIntrinsicsLeft.send(Measurement::CameraIntrinsics(timeStamp, camIntrisics[0]));
	m_outIntrinsicsRight.send(Measurement::CameraIntrinsics(timeStamp, camIntrisics[1]));

	// DEBUG START
	// Write in fixed file OpenCV style
	cv::FileStorage fs("C:\\Users\\FAR-Student\\Desktop\\Phil\\CalibDataset\\intrinsics.yml", cv::FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
			"M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
		fs.release();
	}
	else
		LOG4CPP_ERROR(logger, "Error: can not save the intrinsic parameters.");
	// DEBUG END

	cv::Mat R1, R2, P1, P2, Q;
	cv::Rect validRoi[2];

	cv::stereoRectify(cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, R1, R2, P1, P2, Q,
		cv::CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

	// Save extrinsic parameters
	Math::Pose poseCam1ToCam2 = Math::Pose(Math::Quaternion(Math::Matrix<double, 3, 3>((double*)R.data)), Math::Vector3d((double*)T.data));
	m_outRelativePose.send(Measurement::Pose(timeStamp, poseCam1ToCam2));

	// DEBUG START
	// Write in fixed file OpenCV style
	fs.open("C:\\Users\\FAR-Student\\Desktop\\Phil\\CalibDataset\\extrinsics.yml", cv::FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
		fs.release();
	}
	else
		LOG4CPP_ERROR(logger, "Error: can not save the extrinsic parameters.");
	// DEBUG END
}

} } // namespace Ubitrack::Driver

UBITRACK_REGISTER_COMPONENT(Dataflow::ComponentFactory* const cf) {
	cf->registerComponent< Ubitrack::Drivers::StereoCameraCalibrationOpenCVComponent >("StereoCameraCalibrationOpenCVComponent");
}

