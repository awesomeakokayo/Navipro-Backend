import os
import hashlib
import zlib
import json
from uuid import uuid4
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
import re
from sqlalchemy import create_engine, Column, String, Integer, Boolean, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

load_dotenv()


GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# Use DATABASE_URL from environment, fallback to SQLite if not provided
if DATABASE_URL:
    SQLALCHEMY_DATABASE_URL = DATABASE_URL
else:
    SQLALCHEMY_DATABASE_URL = "sqlite:///./navi.db"

# Create database engine
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class User(Base):
    __tablename__ = "users"
    
    user_id = Column(String, primary_key=True, index=True)
    goal = Column(String)
    target_role = Column(String)
    timeframe = Column(String)
    hours_per_week = Column(String)
    learning_style = Column(String)
    learning_speed = Column(String)
    skill_level = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
class Roadmap(Base):
    __tablename__ = "roadmaps"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    roadmap_data = Column(JSON)  # Stores the entire roadmap structure
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Progress(Base):
    __tablename__ = "progress"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    current_day = Column(Integer, default=1)
    current_week = Column(Integer, default=1)
    current_month = Column(Integer, default=1)
    total_tasks_completed = Column(Integer, default=0)
    start_date = Column(DateTime, default=datetime.utcnow)

class ChatHistory(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    user_message = Column(Text)
    assistant_response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
    
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://naviprototype.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from typing import Optional, List

class ChatMessage(BaseModel):
    message: str
    user_id: str

class TaskCompletion(BaseModel):
    task_completed: bool = True
    task_id: Optional[str] = None

class FullPipelineReq(BaseModel):
    goal: str
    target_role: Optional[str] = ""
    timeframe: str
    hours_per_week: Optional[str] = "10"
    learning_style: Optional[str] = "visual"
    learning_speed: Optional[str] = "average"
    skill_level: Optional[str] = "beginner"


def regroup_by_year(flat_months: list[dict], months_per_year: int = 12) -> list[dict]:
    years = []
    for i in range(0, len(flat_months), months_per_year):
        year_index = i // months_per_year + 1
        months_chunk = flat_months[i:i + months_per_year]
        years.append({
            "year": year_index,
            "months": months_chunk
        })
    return years

# regroup flat weeks into months
def regroup_by_month(flat_weeks: list[dict], weeks_per_month: int = 4) -> list[dict]:
    months = []
    for i in range(0, len(flat_weeks), weeks_per_month):
        month_index = i // weeks_per_month + 1
        weeks_chunk = flat_weeks[i:i + weeks_per_month]
        months.append({
            "month": month_index,
            "weeks": weeks_chunk
        })
    return months

#LLM-based Roadmap Generation
def llm_generate_roadmap(req: FullPipelineReq) -> dict:
    """Generate comprehensive roadmap with weekly focuses and daily tasks"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            print(f"Generating roadmap for: {req.goal}")
            
            if not GROQ_API_KEY:
                raise HTTPException(500, "GROQ_API_KEY not configured")
            
            system_prompt = """You are Navi, a very realistic and practical expert career strategist AI that creates detailed learning roadmaps.
    
    CRITICAL JSON STRUCTURE RULES:
    1. Every month MUST have exactly 4 weeks
    2. Every week MUST have exactly 6 daily tasks
    3. The roadmap structure MUST be:
    {
        "roadmap": [
            {
                "month": 1,
                "month_title": "Specific Month Title",
                "weeks": [
                    {
                        "week": 1,
                        "week_number": 1,
                        "focus": "Specific Week Focus",
                        "daily_tasks": [
                            {
                                "day": 1,
                                "title": "Specific Task Title",
                                "description": "Detailed task description"
                            },
                            // ... exactly 6 tasks per week
                        ]
                    },
                    // ... exactly 4 weeks per month
                ]
            },
            // ... one object per month based on timeframe
        ]
    }
    
    CONTENT REQUIREMENTS:
    1. NO generic titles like "Week 4 Learning" or "Day 1 Task"
    2. Each month_title must describe the learning focus (e.g., "Frontend Framework Mastery")
    3. Each week.focus must be specific (e.g., "React Components and Props")
    4. Each task must be actionable and include a resource
    
    Example of CORRECT content:
    {
        "month_title": "JavaScript Fundamentals",
        "weeks": [{
            "focus": "DOM Manipulation",
            "daily_tasks": [{
                "title": "Learn querySelector Methods",
                "description": "Complete MDN's DOM manipulation tutorial section on querySelectorAll"
            }]
        }]
    }
    """
    
            # Update user prompt to be explicit about months
            timeframe_map = {
                "3_months": "3 months",
                "6_months": "6 months",
                "1_year": "12 months",
                "not_sure": "3 months"  # Explicitly state 3 months
            }
            actual_timeframe = timeframe_map.get(req.timeframe, "3 months")
            
            user_prompt = f"""Create a {actual_timeframe} roadmap for:
    
    Goal: {req.goal}
    Target Role: {req.target_role}
    Available Time: {req.hours_per_week} hours per week
    Learning Style: {req.learning_style}
    Learning Speed: {req.learning_speed}
    Skill Level: {req.skill_level}
    
    IMPORTANT: The roadmap MUST contain exactly {actual_timeframe.split()[0]} months of content.
    Each week should have exactly 6 daily tasks to match the UI template."""
    
            # Add verify=False for testing only
            with httpx.Client(timeout=120.0, verify=False) as client:
                response = client.post(
                    f"{GROQ_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-r1-distill-llama-70b",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 1,
                        "max_tokens": 20000
                    }
                )
            
            # Get raw content
            response_data = response.json()
            if "choices" not in response_data or not response_data["choices"]:
                raise ValueError("Invalid API response: missing choices")
                
            raw_content = response_data["choices"][0]["message"]["content"].strip()
            print("Raw LLM response length:", len(raw_content))

            # Clean and parse JSON
            cleaned_json = clean_llm_response(raw_content)
            roadmap_data = safe_json_loads(cleaned_json)
            
            # Validate structure with timeframe check
            if validate_roadmap_structure(roadmap_data, req):
                return roadmap_data
            else:
                print(f"Attempt {attempt + 1}: Invalid roadmap structure, retrying...")
                continue
                
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise
    
    raise ValueError("Failed to generate valid roadmap after multiple attempts")

def validate_roadmap_structure(roadmap_data: dict, req: FullPipelineReq) -> bool:
    """Validate that the roadmap has the correct structure and month count"""
    try:
        roadmap = roadmap_data.get("roadmap", [])
        
        # Verify correct number of months based on timeframe
        timeframe_map = {
            "3_months": 3,
            "6_months": 6,
            "1_year": 12,
            "not_sure": 3  # Enforcing 3 months for "not sure"
        }
        expected_months = timeframe_map.get(req.timeframe, 3)
        
        if len(roadmap) != expected_months:
            print(f"Expected {expected_months} months, got {len(roadmap)} months")
            return False
        
        # Check each month
        for month in roadmap:
            if len(month.get("weeks", [])) != 4:
                print(f"Month {month.get('month')} doesn't have exactly 4 weeks")
                return False
                
            # Check each week
            for week in month["weeks"]:
                if len(week.get("daily_tasks", [])) != 6:
                    print(f"Week {week.get('week')} in month {month.get('month')} doesn't have exactly 6 tasks")
                    return False
                    
                # Check for generic content
                if "Learning" in week.get("focus", ""):
                    print(f"Generic week focus detected: {week.get('focus')}")
                    return False
                    
                # Check tasks
                for task in week["daily_tasks"]:
                    if "Task" in task.get("title", "") or "Day" in task.get("title", ""):
                        print(f"Generic task title detected: {task.get('title')}")
                        return False
        
        return True
    except Exception as e:
        print(f"Validation error: {str(e)}")
        return False

def safe_json_loads(raw_content: str) -> dict:
    """Safely parse JSON content with multiple fallback attempts"""
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError as e:
        print("Initial JSON parsing failed, attempting repairs...")
        try:
            # Try fixing common JSON issues
            fixed = raw_content
            # Replace single quotes with double quotes
            fixed = re.sub(r"(?<!\\)'", '"', fixed)
            # Remove trailing commas
            fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
            # Ensure property names are quoted
            fixed = re.sub(r'([{,]\s*)(\w+)(:)', r'\1"\2"\3', fixed)
            
            print("Attempting to parse fixed JSON...")
            return json.loads(fixed)
        except Exception as repair_err:
            print("JSON repair failed:", str(repair_err))
            print("Problematic JSON:", raw_content[:200])
            raise ValueError(f"Could not parse JSON response: {str(e)}")

def clean_llm_response(raw_content: str) -> str:
    """Clean and extract valid JSON from LLM response"""
    try:
        # Remove markdown code blocks
        if "```json" in raw_content:
            parts = raw_content.split("```json")
            raw_content = parts[1].split("```")[0]
        
        # Find the outermost JSON object
        json_start = raw_content.find('{')
        json_end = raw_content.rfind('}') + 1
        
        if json_start == -1 or json_end <= json_start:
            raise ValueError("No valid JSON object found in response")
        
        cleaned_json = raw_content[json_start:json_end]
        
        # Remove any trailing commas before } or ]
        cleaned_json = re.sub(r',(\s*[}\]])', r'\1', cleaned_json)
        
        return cleaned_json.strip()
    except Exception as e:
        print(f"Error cleaning JSON: {str(e)}")
        print("Raw content:", raw_content[:200])  # Print first 200 chars for debugging
        raise

def enhance_roadmap_structure(roadmap_data: dict) -> dict:
    """Add IDs, completion status, and metadata to roadmap"""

    #Start progress tracking
    roadmap_data["progress"] = {
        "current_day": 1,
        "current_week": 1,
        "current_month": 1,
        "total_tasks_completed": 0,
        "start_date": datetime.now().isoformat()
    }

    #Add IDs and completion status
    for month in roadmap_data.get("roadmap", []):
        month_num = month["month"]

        for week in month.get("weeks", []):
            week_num = week["week"]
            week["week_id"] = f"month_{month_num}_week_{week_num}"
            week["completed"] = False

            for task in week.get("daily_tasks", []):
                day_num = task["day"]
                task["task_id"] = f"m{month_num}_w{week_num}_d{day_num}"
                task["completed"] = False
                task["completed_date"] = None

                # Add motivational elements if missing
                if "description" not in task:
                    task["description"] = f"Master the fundamentals of {week['focus']}"
                if "estimated_time" not in task:
                    task["estimated_time"] = "2 hours"
    return roadmap_data

def create_fallback_roadmap(req: FullPipelineReq) -> dict:
    """Create a fallback roadmap when LLM generation fails"""
    print("Creating fallback roadmap")
    
    # Convert timeframe to months
    timeframe_map = {
        "3_months": 3,
        "6_months": 6,
        "1_year": 12,
        "not_sure": 3
    }
    
    months = timeframe_map.get(req.timeframe, 3)
    
    roadmap = {
        "goal": req.goal,
        "target_role": req.target_role,
        "timeframe": req.timeframe,
        "learning_speed": req.learning_speed,
        "skill_level": req.skill_level,
        "roadmap": []
    }
    
    for month_num in range(1, months + 1):
        month_data = {
            "month": month_num,
            "month_title": f"Month {month_num} Focus",
            "weeks": []
        }
        
        # Create 4 weeks per month
        for week_num in range(1, 5):
            week_data = {
                "week": week_num,
                "week_number": week_num,
                "focus": f"Week {week_num} Learning",
                "daily_tasks": []
            }
            
            # Create 6 daily tasks per week
            for day_num in range(1, 7):
                task_data = {
                    "day": day_num,
                    "title": f"Day {day_num} Task",
                    "description": f"Complete day {day_num} learning activities"
                }
                week_data["daily_tasks"].append(task_data)
            
            month_data["weeks"].append(week_data)
        
        roadmap["roadmap"].append(month_data)
    
    return roadmap

def normalize_roadmap_structure(roadmap_data: dict, months: int = 3, weeks_per_month: int = 4, days_per_week: int = 6) -> dict:
    """
    Ensures the roadmap has complete, relevant content
    """
    # Fill missing months
    roadmap = roadmap_data.get("roadmap", [])
    for month_num in range(1, months + 1):
        # Find or create month
        month = next((m for m in roadmap if m.get("month") == month_num), None)
        if not month:
            month = {
                "month": month_num,
                "month_title": f"Month {month_num} Focus",
                "weeks": []
            }
            roadmap.append(month)
        # Fill missing weeks
        for week_num in range(1, weeks_per_month + 1):
            week = next((w for w in month["weeks"] if w.get("week") == week_num), None)
            if not week:
                week = {
                    "week": week_num,
                    "week_number": week_num,
                    "focus": f"Week {week_num} Learning",
                    "daily_tasks": []
                }
                month["weeks"].append(week)
            # Fill missing daily tasks
            for day_num in range(1, days_per_week + 1):
                if not any(t.get("day") == day_num for t in week["daily_tasks"]):
                    week["daily_tasks"].append({
                        "day": day_num,
                        "title": f"Day {day_num} Task",
                        "description": f"Complete day {day_num} learning activities"
                    })
            # Sort daily tasks
            week["daily_tasks"].sort(key=lambda t: t["day"])
        # Sort weeks
        month["weeks"].sort(key=lambda w: w["week"])
    # Sort months
    roadmap.sort(key=lambda m: m["month"])
    roadmap_data["roadmap"] = roadmap
    return roadmap_data


# DAILY TASK SYSTEM
def get_current_daily_task(user_id: str, db: Session) -> dict:
    """Get the current daily task for the user with motivation"""

    progress = db.query(Progress).filter(Progress.user_id == user_id).first()
    if not progress:
        return {}
    
    # Get user roadmap
    roadmap_record = db.query(Roadmap).filter(Roadmap.user_id == user_id).first()
    if not roadmap_record:
        return {}
    
    roadmap = roadmap_record.roadmap_data
    
    current_month = progress.current_month
    current_week = progress.current_week
    current_day = progress.current_day

    # find current task
    for month in roadmap.get("roadmap", []):
        if month["month"] == current_month:
            for week in month["weeks"]:
                if week["week"] == current_week:
                    for task in week.get("daily_tasks", []):
                        if task["day"] == current_day and not task.get("completed", False):

                            #Generate motivational message
                            motivation = generate_motivational_message(
                                roadmap["goal"],
                                task["title"],
                                progress.total_tasks_completed
                            )

                            return {
                                    "task_id": task["task_id"],
                                    "title": task["title"],
                                    "description": task.get("description", ""),
                                    "goal": roadmap.get("goal", ""),                 # <-- FIXED
                                    "estimated_time": task.get("estimated_time", ""),# <-- guard
                                    "resources": task.get("resources", []),
                                    "week_focus": week.get("focus", ""),
                                    "motivation_message": motivation,
                                    "progress": {
                                        "current_day": current_day,
                                        "current_week": current_week,
                                        "current_month": current_month,
                                        "total_completed": progress.total_tasks_completed 
                                    }
                                }

    return {"message": "All tasks completed! 🎉"}

def generate_motivational_message(goal: str, task_title: str, completed_task: int) -> str:
    messages = [
        f"🚀 Great job! You're {completed_task} steps closer to '{goal}'. Every expert was once a beginner!",
        f"💪 You're building something amazing! The '{task_title}' task is a crucial building block for '{goal}'.",
        f"🌟 Remember why you started: '{goal}'. Today's task brings you closer to that dream!",
        f"🔥 Consistency beats perfection! You've completed {completed_task} tasks already. Keep the momentum going!",
        f"💡 Every practice, every concept learned, every task completed is an investment in your future self!",
        f"🎯 Focus on progress, not perfection. '{task_title}' might seem small, but it's a vital step toward '{goal}'.",
        f"⭐ You're not just learning — you're building a new future. '{goal}' is within reach!",
        f"🚗 You don't need to see the whole road — just the next turn. Today's task: '{task_title}'."
    ]
    import random
    return random.choice(messages)


def mark_task_completed(user_id: str, task_id: str, db: Session) -> dict:
    """Mark current task as completed and move to next"""

    # Get user progress and roadmap
    progress = db.query(Progress).filter(Progress.user_id == user_id).first()
    roadmap_record = db.query(Roadmap).filter(Roadmap.user_id == user_id).first()
    
    if not progress or not roadmap_record:
        return {"error": "User not found"}
    
    roadmap = roadmap_record.roadmap_data
    
    # Find and mark task as completed
    task_found = False
    for month in roadmap.get("roadmap", []):
        for week in month["weeks"]:
            for task in week.get("daily_tasks", []):
                if task["task_id"] == task_id:
                    task["completed"] = True
                    task["completed_date"] = datetime.now().isoformat()
                    task_found = True

                    #Update progress
                    progress.total_tasks_completed += 1

                    #Move to next day
                    advance_to_next_task(roadmap, progress)

                    # Save updated roadmap and progress to database
                    roadmap_record.roadmap_data = roadmap
                    db.commit()

                    return {
                        "status": "success",
                        "message": "Task completed! 🎉",
                        "completed_task": task["title"],
                        "total_completed": progress.total_tasks_completed
                    }
    if not task_found:
        return{"error": "Task not found"}
    
def advance_to_next_task(roadmap: dict, progress: Progress):
    """Move user to the next task/week/month"""
    current_month = progress.current_month
    current_week = progress.current_week
    current_day = progress.current_day

    #find next incomplete task
    for month in roadmap.get("roadmap", []):
        if month["month"] >= current_month:
            for week in month['weeks']:
                if (month["month"] > current_month) or (week["week"] >= current_week):
                    for task in week.get("daily_tasks", []):
                        if ((month["month"] > current_month) or (week["week"] > current_week) or (task["day"] > current_day)) and not task.get("completed", False):

                            progress.current_month = month["month"]
                            progress.current_week = week["week"]
                            progress.current_day = task["day"]
                            return
    #If no more tasks, mark as completed
    progress.current_day = -1  #to indicate completion

# YOUTUBE VIDEO RECOMMENDATION
def get_current_week_videos(user_id: str, db: Session) -> dict:
    """Get Youtube videos for current week's focus"""

     # Get user progress and roadmap
    progress = db.query(Progress).filter(Progress.user_id == user_id).first()
    roadmap_record = db.query(Roadmap).filter(Roadmap.user_id == user_id).first()
    
    if not progress or not roadmap_record:
        return {"error": "User not found"}

    roadmap = roadmap_record.roadmap_data
    current_month = progress.current_month
    current_week = progress.current_week

    # Create a request object from stored roadmap data
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        return {"error": "User not found"}
    
    req = FullPipelineReq(
        goal=roadmap.get("goal", ""),
        target_role=roadmap.get("target_role", ""),
        timeframe=roadmap.get("timeframe", "3_months"),
        learning_style=roadmap.get("learning_style", "visual"),
        learning_speed=roadmap.get("learning_speed", "average"),
        skill_level=roadmap.get("skill_level", "beginner")
    )

    # Find and extract current week focus
    current_week_focus = None
    for month in roadmap.get("roadmap", []):
        if month["month"] == current_month:
            for week in month["weeks"]:
                if week["week"] == current_week:
                    current_week_focus = week["focus"]
                    break

    if not current_week_focus:
        return {"error": "Current week not found"}
    
    videos = search_youtube_videos(current_week_focus, req)

    return {
        "week_focus": current_week_focus,
        "week_info": f"Month {current_month}, Week {current_week}",
        "videos": videos,
        "total_videos": len(videos)
    }

def search_youtube_videos(query: str, req: FullPipelineReq, max_results: int = 8) -> list:
    """Search Youtube for target role videos"""

    if not YOUTUBE_API_KEY:
        print("No Youtube API key - returning sample videos")
        return get_sample_videos(query)
    
    try:
        enhanced_query = f"{query} {req.target_role} tutorial coding"

        with httpx.Client(timeout=30.0) as client:
            #search videos
            search_response = client.get(
                "https://www.googleapis.com/youtube/v3/search",params={
                    "key": YOUTUBE_API_KEY,
                    "part": "snippet",
                    "q": enhanced_query,
                    "type": "video",
                    "maxResults": max_results,
                    "order": "relevance",
                    "videoDuration": "medium"
                }
            )

            if not search_response.is_success:
                return get_sample_videos(query)
            
            search_data = search_response.json()
            video_ids = [item["id"] ["videoId"] for item in search_data.get("items", [])]

            if not video_ids:
                return get_sample_videos(query)
            
            #Get video details
            details_response = client.get(
                "https://www.googleapis.com/youtube/v3/videos", params={
                    "key": YOUTUBE_API_KEY,
                    "part": "snippet,contentDetails,statistics",
                    "id": ",".join(video_ids)
                }
            )

            if not details_response.is_success:
                return get_sample_videos(query)
            
            details_data = details_response.json()

            videos = []
            for item in details_data.get("items", []):
                video = {
                    "title": item["snippet"]["title"],
                    "url": f"https://www.youtube.com/watch?v={item['id']}",
                    "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
                    "channel": item["snippet"]["channelTitle"],
                    "duration": item["contentDetails"]["duration"],
                    "views": item["statistics"].get("viewCount", "0")
                }
                videos.append(video)

            #Sort by views
            videos.sort(key=lambda x: int(x["views"]), reverse=True)
            return videos[:6]
    
    except Exception as e:
        print(f"Youtube API error: {e}")
        return get_sample_videos(query)

def get_sample_videos(query: str) -> list:
    """Return sample videos when Youtube API is not available"""
    return [
        {
            "title": f"{query} - Complete Tutorial",
            "url": "https://youtube.com/watch?v=sample1",
            "thumbnail": "https://i.ytimg.com/vi/sample/mqdefault.jpg",
            "channel": "Programming Tutorial",
            "duration": "PT15M30S",
            "views": "150000",
            "description": f"Complete tutorial on {query} for beginners..."
        },
        {
            "title": f"Learn {query} in 20 Minutes",
            "url": "https://youtube.com/watch?v=sample2",
            "thumbnail": "https://i.ytimg.com/vi/sample/mqdefault.jpg",
            "channel": "Code Academy",
            "duration": "PT20M15S",
            "views": "89000",
            "description": f"Quick crash course on {query}..."
        }
    ]

# CHATBOT SYSTEM
def get_ai_chat_response(user_id: str, message: str, db: Session) -> dict:
    """Generate AI chat response based on user's roadmap context"""

    # Get user data
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        return {"error": "User not found"}
    
    # Get user progress
    progress = db.query(Progress).filter(Progress.user_id == user_id).first()
    total_completed = progress.total_tasks_completed if progress else 0

    # Get chat history from database
    chat_history_records = db.query(ChatHistory).filter(
        ChatHistory.user_id == user_id
    ).order_by(ChatHistory.timestamp.desc()).limit(6).all()
    
    # Reverse to get chronological order
    chat_history_records.reverse()

    #Build AIs context-aware prompt
    system_prompt = f"""You are Navi, a helpful career mentor AI assitant
    
    Context about the user:
    - Career Goal: {user.goal}
    - Tasks Completed: {total_completed}
    - Learning Journey: Currently working on their {user.target_role} roadmap
    
    Your role:
    1. Answer questions about {user.target_role}, career development, and learning
    2. Provide encouragement and motivation
    3. Give practical advice based on their goal
    4. Keep responses conversational and supportive
    5. If asked about progress, reference their completed tasks
    
    Guidelines:
    - Be encouraging and positive
    - Provide practical, actionable advice
    - Keep responses under 200 words unless more detail is needed
    - Reference their goal when relevant
    - Do not use asterisks * to format words or actions. Write all the text as plain sentences in full paragraphs with proper spacing. No markdown (DO NOT USE MARKDOWN FORMATTING), no special characters for emphasis. keep everything simple and clean for easy readability
    - Break ideas into paragraphs by adding a blank line between them.
    - keep sentences spaced properly for readability
    """
    
    # Prepare conversation history
    messages = [{"role": "system", "content": system_prompt}]

    # Add recent chat history (last  6 messages to stay within token limits)
    for record in chat_history_records:
        messages.append({"role": "user", "content": record.user_message})
        messages.append({"role": "assistant", "content": record.assistant_response})

    # Add current message
    messages.append({"role": "user", "content": message})

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{GROQ_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-oss-120b",
                    "messages": messages,
                    "temperature": 1,
                    "max_tokens": 20000
                }
            )
        response.raise_for_status()
        ai_response = response.json()["choices"][0]["message"]["content"]

       # Store in database
        chat_record = ChatHistory(
            user_id=user_id,
            user_message=message,
            assistant_response=ai_response
        )
        db.add(chat_record)
        db.commit()
        
        return {
            "response": ai_response,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Chat error: {e}")
        return {
            "response": "I'm having trouble connecting right now. Please try again in a moment!",
            "timestamp": datetime.now().isoformat()
        }

# API Endpoints

@app.post("/api/generate_roadmap")
def api_generate_roadmap(req: FullPipelineReq, db: Session = Depends (get_db)):
    """Generate initial roadmap"""
    try:
        # Generate roadmap with validation
        roadmap = llm_generate_roadmap(req)

        timeframe_map = {
            "3_months": 3,
            "6_months": 6,
            "1_year": 12,
            "not_sure": 3  # Make sure it's consistent
        }
        
        # Add user data to roadmap
        roadmap.update({
            "goal": req.goal,
            "target_role": req.target_role,
            "timeframe": req.timeframe,
            "learning_style": req.learning_style,
            "learning_speed": req.learning_speed,
            "skill_level": req.skill_level
        })
        
        # Add IDs and metadata
        roadmap = enhance_roadmap_structure(roadmap)
        
        # Store for user
        user_id = str(uuid4())

        user = User(
            user_id=user_id,
            goal=req.goal,
            target_role=req.target_role,
            timeframe=req.timeframe,
            hours_per_week=req.hours_per_week,
            learning_style=req.learning_style,
            learning_speed=req.learning_speed,
            skill_level=req.skill_level
        )
        db.add(user)
        
        # Create roadmap record
        roadmap_record = Roadmap(
            user_id=user_id,
            roadmap_data=roadmap
        )
        db.add(roadmap_record)

        # Create progress record
        progress = Progress(
            user_id=user_id,
            current_day=1,
            current_week=1,
            current_month=1,
            total_tasks_completed=0
        )
        db.add(progress)
        
        db.commit()

        
        # Print user ID clearly in terminal
        print("\n" + "="*50)
        print(f"Generated User ID: {user_id}")
        print(f"Goal: {req.goal}")
        print(f"Target Role: {req.target_role}")
        print("="*50 + "\n")
        
        return {
            "success": True,
            "user_id": user_id,
            "roadmap": roadmap,
            "message": "Roadmap generated successfully!"
        }
    except Exception as e:
        print(f"Roadmap generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate roadmap: {str(e)}"
        )

@app.get("/api/user_roadmap/{user_id}")
def api_get_user_roadmap(user_id: str, db: Session = Depends(get_db)):
    """Get user's roadmap from database"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    
    roadmap_record = db.query(Roadmap).filter(Roadmap.user_id == user_id).first()
    if not roadmap_record:
        raise HTTPException(404, "Roadmap not found")
    
    progress = db.query(Progress).filter(Progress.user_id == user_id).first()
    
    roadmap_data = roadmap_record.roadmap_data
    
    # Add progress information
    if progress:
        roadmap_data["progress"] = {
            "current_day": progress.current_day,
            "current_week": progress.current_week,
            "current_month": progress.current_month,
            "total_tasks_completed": progress.total_tasks_completed,
            "start_date": progress.start_date.isoformat() if progress.start_date else None
        }
    
    return roadmap_data
    
@app.get("/api/daily_task/{user_id}")
def api_get_daily_task(user_id:str, db: Session = Depends(get_db)):
    """Get current daily task with motivation"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    
    task = get_current_daily_task(user_id, db)
    if not task:
        raise HTTPException(404, "No current task found")
    
    return task

@app.post("/api/complete_task/{user_id}")
def api_complete_task(user_id:str, completion: TaskCompletion, db: Session = Depends(get_db)):
    """Mark current task as completed If task_id is provided, mark that task; otherwise use the current daily task."""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    
     # If client provided a task_id, use it; otherwise use current daily task
    target_task_id = completion.task_id
    if not target_task_id:
        current_task = get_current_daily_task(user_id, db)
        if not current_task or "task_id" not in current_task:
            raise HTTPException(404, "No current task to complete")
        target_task_id = current_task["task_id"]

    result = mark_task_completed(user_id, target_task_id, db)

    if "error" in result:
        raise HTTPException(400, result["error"])

    # Fetch updated roadmap and progress to return to client for immediate UI sync
    roadmap_record = db.query(Roadmap).filter(Roadmap.user_id == user_id).first()
    progress = db.query(Progress).filter(Progress.user_id == user_id).first()

    updated_roadmap = roadmap_record.roadmap_data if roadmap_record else {}
    if progress:
        updated_roadmap["progress"] = {
            "current_day": progress.current_day,
            "current_week": progress.current_week,
            "current_month": progress.current_month,
            "total_tasks_completed": progress.total_tasks_completed,
            "start_date": progress.start_date.isoformat() if progress.start_date else None
        }

    # return original result plus the fresh snapshot
    response_payload = {
        **result,
        "roadmap": updated_roadmap,
        "progress": updated_roadmap.get("progress", {})
    }

    return response_payload


@app.get("/api/week_videos/{user_id}")
def api_get_week_videos(user_id: str, db: Session = Depends(get_db)):
    """Get Youtube videos for current week"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    
    videos = get_current_week_videos(user_id, db)
    if "error" in videos:
        raise HTTPException(404, videos["error"])
    
    return videos

@app.post("/api/chat/{user_id}")
async def api_chat(user_id: str, chat_msg: ChatMessage, db: Session = Depends(get_db)):
    """Chat with AI assistant"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")

    response = get_ai_chat_response(user_id, chat_msg.message, db) 
    if "error" in response:
        raise HTTPException(400, response["error"])
    
    return response

@app.get("/api/user_progress/{user_id}")
def api_get_user_progress(user_id: str, db: Session = Depends(get_db)):
    """Get user's overall progress"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    
    progress = db.query(Progress).filter(Progress.user_id == user_id).first()
    roadmap_record = db.query(Roadmap).filter(Roadmap.user_id == user_id).first()
    
    if not progress or not roadmap_record:
        raise HTTPException(404, "Progress data not found")

    roadmap = roadmap_record.roadmap_data

    # Calculate completion percentage
    total_tasks = 0
    completed_tasks = progress.total_tasks_completed

    for month in roadmap.get("roadmap", []):
        for week in month["weeks"]:
            total_tasks += len(week.get("daily_tasks", []))
    
    completion_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0

    return {
        "goal": user.goal,
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "completion_percentage": round(completion_percentage, 1),
        "current_month": progress.current_month,
        "current_week": progress.current_week,
        "current_day": progress.current_day,
        "start_date": progress.start_date.isoformat() if progress.start_date else None
    }

@app.get("/api/health")
def health_check(db: Session = Depends(get_db)):
    user_count = db.query(User).count()
    return {
        "status": "healthy",
        "active_users": user_count,
        "groq_configured": bool(GROQ_API_KEY),
        "youtube_configured": bool(YOUTUBE_API_KEY)
    }

# Legacy endpoint for compatibility
@app.post("/api/full_pipeline")
def api_full_pipeline(req: FullPipelineReq, db: Session = Depends(get_db)):
    """Legacy endpoint - redirects to new generate_roadmap"""
    return api_generate_roadmap(req, db)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("agent_orchestra:app", host="0.0.0.0", port=port)



