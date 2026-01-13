SceneForge 🚀

AI-Powered Image & Video to 3D Reconstruction Platform

📌 Overview

SceneForge is a full-stack AI application that converts real-world images and short videos into usable 3D assets (.glb / .obj) using computer vision and modern backend systems.

The project focuses on automating early-stage 3D asset generation for AR/VR, game development, ML dataset creation, and rapid prototyping.

⚠️ Current Status: Actively under development
Core functionality is implemented and working, with approximately 60% of the planned pipeline completed.

🎯 Problem Statement

Manually creating 3D assets from real-world references is time-consuming, skill-intensive, and often inaccessible to non-3D artists.

SceneForge aims to:

Reduce manual modeling effort

Automate geometry reconstruction

Provide fast, draft-quality 3D assets suitable for iteration and prototyping

⚙️ Current Capabilities (Implemented)
✅ Input Support

Image upload

Short video upload (10–30 seconds recommended)

✅ AI Processing Pipeline (Working)

Depth estimation from images/video frames

Foreground segmentation (object isolation)

Multi-frame geometry fusion (partial)

Draft 3D mesh generation

Export to .glb / .obj

Preview assets via the web interface

✅ Backend Architecture

FastAPI-based REST API

Asynchronous request handling

Background task processing (Celery + Redis)

File upload & processing pipeline

Deployed backend (Render)

✅ Frontend

Modern web UI

File upload & progress handling

Result preview and download

Deployed on Vercel

🚧 Current Limitations (Known & Expected)

SceneForge currently generates draft-quality 3D assets, not final production models.

Limitations include:

Geometry artifacts on complex or reflective objects

Reduced accuracy for transparent or thin objects (e.g., glass bottles)

Partial texture quality

Limited handling of complex camera motion in videos

These limitations are inherent to early-stage automated reconstruction pipelines and are being actively addressed.

🔮 Planned Improvements (Upcoming Work)

The following enhancements are planned to move from draft-quality (~60%) to production-quality (~90%) outputs:

Robust camera pose estimation (SfM)

Improved multi-view volumetric fusion

Geometry priors for object stabilization

Advanced texture projection

Hybrid AI fallback (generative 3D for hard objects)

Quality scoring & automatic pipeline routing

Improved UI/UX and real-time 3D preview

Authentication & user project history

🧠 Design Philosophy

SceneForge is designed as a progressive, modular pipeline:

Fast draft first → refine iteratively

The system prioritizes:

Automation over manual steps

Transparency of limitations

Iterative improvement

Real-world applicability

🏗️ Tech Stack
Frontend

React

Modern UI framework

Vercel deployment

Backend

Python

FastAPI

Celery + Redis

Async processing

Render deployment

AI / 3D

Computer Vision

Depth estimation

Multi-view geometry

3D mesh processing

GLB / OBJ export

🌱 Project Status

This project is actively evolving.

Current outputs demonstrate:

Functional end-to-end flow

Working AI pipeline

Realistic intermediate results

Strong foundation for future refinement

Contributions, feedback, and suggestions are welcome.

⚠️ Deployment & Infrastructure Limitations

SceneForge is currently deployed using Render’s free-tier infrastructure, which imposes strict limits on disk space, memory, and compute resources.

Due to these constraints:

Advanced AI models and heavier 3D reconstruction pipelines cannot be fully executed in the deployed environment

Large model weights and multi-stage processing pipelines exceed the available storage and runtime limits

Some high-quality outputs (shown in the demo images/screenshots) are generated locally during development, where full resources are available

As a result:

The live deployed version demonstrates the working pipeline and architecture, but runs in a resource-constrained mode

Certain advanced reconstruction steps are intentionally disabled or simplified in production deployment

🚀 Note: Running SceneForge with advanced models, higher accuracy pipelines, and full multi-view reconstruction requires a premium cloud plan or dedicated infrastructure with higher storage and compute capacity.

🔮 Planned Infrastructure Upgrade

Future iterations of SceneForge aim to:

Deploy on higher-tier infrastructure (GPU-enabled instances)

Enable advanced depth, fusion, and generative models in production

Support larger uploads and more complex scenes

Deliver higher-quality 3D assets directly from the live website

📌 Transparency Statement

The current deployment reflects real-world engineering trade-offs between cost, performance, and accessibility. The project prioritizes demonstrating system design, pipeline correctness, and scalability potential, with infrastructure upgrades planned as the next phase.

📬 Contact

Author: Abhishek Kamthe
Project: SceneForge
Status: Active development

⭐ Note for Reviewers & Recruiters

SceneForge demonstrates system design, applied AI, and full-stack engineering rather than a polished final product. The project intentionally exposes intermediate results to reflect real-world AI development workflows.

