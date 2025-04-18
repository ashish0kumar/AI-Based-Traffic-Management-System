# AI-Based-Traffic-Management-System
AI-Based Traffic Management System Using Computer Vision and Genetic Algorithms

## Overview

The Smart Adaptive Traffic Management System leverages AI, computer vision and genetic algorithms to optimize traffic flow at intersections. This system analyzes vehicle counts from video feeds using YOLOv4, processes the data, and adjusts traffic signal timings using a genetic algorithm to improve traffic flow.

## Features
- Vehicle Detection: Uses YOLOv4 for real-time vehicle detection from video feeds.
- Traffic Optimization: Employs a genetic algorithm to determine optimal green light times based on vehicle counts.
- Web Interface: Allows users to upload traffic videos, view processing results, and receive optimized traffic management recommendations.

## Getting Started

### Prerequisites

- Python 3.x
- Nodejs
- OpenCV
- YOLOv4 weights and configuration files
- Required Python packages (listed in requirements.txt)

## Local Setup

Clone the repository:

```bash
git clone https://github.com/ashish0kumar/AI-Based-Traffic-Management-System.git
cd AI-Based-Traffic-Management-System
```

Start the backend server:

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Start the frontend server:
```bash
cd frontend
npm install
npm start
```

Upload Traffic Videos: <br/>
Use the web interface to upload 4 traffic videos. The system will process the videos and display optimized green light times based on the analysis. <br>
If you don’t have access to traffic intersection footage, you can use the 4 example videos available in the `backend/uploads/` folder.
