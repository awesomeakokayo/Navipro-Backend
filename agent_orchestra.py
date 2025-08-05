import os
import json
from uuid import uuid4
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

load_dotenv()


GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# In-memory user store (for demo)
user_store: dict[str, dict] = {}
chat_history: dict[str, list] = {}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from typing import Optional, List

class ChatMessage(BaseModel):
    message: str
    user_id: str

class TaskCompletion(BaseModel):
    task_completed: bool = True

class FullPipelineReq(BaseModel):
    goal: str
    target_role: Optional[str] = ""
    why: str
    timeframe: str
    hours_per_week: Optional[str] = "10"
    skills: Optional[list[str]] = []
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
    """Generate comprehensive roadmap with weekly focuses and daily taks"""
    print(f" Generating roadmap for: {req.goal}")
    
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY not configured")
    
    system_prompt = """You are Navi, a very realistic and practical expert career strategist AI that creates detailed learning roadmaps.

Your job is to design a personalized roadmap that fits the user's situation. You must be detailed, realistic, and output in valid JSON.

Create a structured roadmap that breaks down into:
- Monthly phases with clear focus areas
- Weekly focuses with specific learning topics
- Daily task that can be completed in the specified time

Requirements:
1. Think step-by-step like a mentor coaching a student from scratch.
2. Break the roadmap into clear weekly stages based on their learning speed and timeframe.
3. Each week must have a specific "focus" (e.g., "JavaScript DOM Manipulation", "React Hooks", "CSS Flexbox and Grid").
4. Each task chould be achievable in 1-3 hours
5. Each week should have 5 daily tasks.
6. Match tasks and concepts with the user's skill level.
7. Ensure everything can fit within the timeframe realistically.
8. Use only **free resources** (e.g., FreeCodeCamp, Scrimba, MDN, Youtube).
9. Output only valid JSON in the format below.


JSON format:
{
    "goal": "...",
    "why": "...",
    "timeframe": "...",
    "learning_speed": "...",
    "skill_level": "...",
    "roadmap": [
        {
            "month": 1,
            "focus": "Foundation Building",
            "weeks": [
                {
                    "week": 1,
                    "focus": "HTML Fundamentals and Semantic Structure",
                    "daily_tasks": [
                        { "day": 1,
                        "title": "Learn HTML Basics", 
                        "description": "Study HTML elements, tags, and document structure", 
                        "goal": "Undestand how HTML creates web page structure",
                        "estimated_time": "2 hours",
                        "resources": ["MDN HTML Basics", "FreeCodeCamp HTML section"] 
                        }
                    ]
                }
            ]
        }
    ]
}"""

    user_prompt = f"""Create a {req.timeframe} roadmap for:

Goal: {req.goal}
Target Role: {req.target_role}  
Why: {req.why}
Available Time: {req.hours_per_week} hours per week
Current Skills: {', '.join(req.skills) if req.skills else 'None'}
Learning Style: {req.learning_style}
Learning Speed: {req.learning_speed}
Skill Level: {req.skill_level}

Create detailed daily tasks that progressively build skills."""

    try:
        with httpx.Client(timeout=120.0) as client:
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
            "temperature": 0.3,
            "max_tokens": 4000
        }
    )
        
        # Clean the response to extract JSON
        raw_content = raw_content.strip()
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:]
        if raw_content.endswith("```"):
            raw_content = raw_content[:-3]
        raw_content = raw_content.strip()
        
        # Extract JSON
        json_start = raw_content.find('{')
        json_end = raw_content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            raw_content = raw_content[json_start:json_end]
        
        roadmap_data = json.loads(raw_content)

        # Add metadata and IDs
        roadmap_data = enhance_roadmap_structure(roadmap_data)
        
        print("Roadmap generated successfully")
        return roadmap_data
            
    except Exception as e:
            print(f"Error generating roadmap: {e}")
            return create_fallback_roadmap(req)

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
                if "goal" not in task:
                    task["goal"] = f"Master the fundamentals of {week['focus']}"
                if "estimated_time" not in task:
                    task["estimated_time"] = "2 hours"
    return roadmap_data

def create_fallback_roadmap(req: FullPipelineReq) -> dict:
    print("Try again: roadmap could not be generated")

# DAILY TASK SYSTEM
def get_current_daily_task(user_id: str) -> dict:
    """Get the current daily task for the user with motivation"""

    roadmap = user_store.get(user_id)
    if not roadmap:
        return{}
    
    progress = roadmap.get("progress", {})
    current_month = progress.get("current_month", 1)
    current_week = progress.get("current_week", 1)
    current_day = progress.get("current_day", 1)

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
                                progress.get("total_tasks_completed",0)
                            )

                            return {
                                "task_id": task["task_id"],
                                "title": task["title"],
                                "description": task["description"],
                                "goal": task["goal"],
                                "estimated_time": task["estimated_time"],
                                "resources": task.get("resources", []),
                                "week_focus": week["focus"],
                                "motivation_message": motivation,
                                "progress": {
                                    "current_day": current_day,
                                    "current_week": current_week,
                                    "current_month": current_month,
                                    "total_completed": progress.get("total_tasks_completed", 0)
                                }
                            }
    return {"message": "All tasks completed! 🎉"}

def generate_motivational_message(goal: str, task_title: str, completed_task: int) -> str:
    """Generate AI-powered motivational message"""

    messages = [
        f"🚀 Great job! You're {completed_task} steps closer to '{goal}. Every expert was once a beginner!'",
        f"💪 You're building somthing amazing! This '{task_title}' task is a crucial building block for '{goal}'.",
        f" 🌟 Remember why you started: '{goal}'. Today's task brings you closer to that dream!"
        f"🔥 Consistency beats perfection! You've completed {completed_task} tasks already. Keep the momentum going!",
        f" 💡Every practice, every concept learned, every task completed is an investment in your future self!",
        f"🎯 Focus on progress, not perfection. '{task_title}' might seem small, but it's a vital step forward '{goal}'!",
        f"⭐ You're not just learning to code, you're building a new future. '{goal}' is within reach!",
        f"🚗 Think of learning like driving - you don't need to see the whole road, just a few steps. Today's task: '{task_title}' "

    ]

    import random
    return random.choice(messages)

def mark_task_completed(user_id: str, task_id: str) -> dict:
    """Mark current task as completed and move to next"""

    roadmap = user_store.get(user_id)
    if not roadmap:
        return{"error": "User not found"}
    
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
                    progress = roadmap.get("progress", {})
                    progress["total_tasks_completed"] = progress.get("total_tasks_completed", 0) + 1

                    #Move to next day
                    advance_to_next_task(roadmap)

                    return {
                        "status": "success",
                        "message": "Task completed! 🎉",
                        "completed_task": task["title"],
                        "total_completed": progress["total_tasks_completed"]
                    }
    if not task_found:
        return{"error": "Task not found"}
    
def advance_to_next_task(roadmap: dict):
    """Move user to the next task/week/month"""

    progress = roadmap.get("progress", {})
    current_month = progress.get("current_month", 1)
    current_week = progress.get("current_week", 1)
    current_day = progress.get("current_day", 1)

    #find next incomplete task
    for month in roadmap.get("roadmap", []):
        if month["month">= current_month]:
            for week in month['weeks']:
                if (month["month"] > current_month) or (week["week"] >= current_week):
                    for task in week.get("daily_task", []):
                        if ((month["month"] > current_month) or (week["week"] > current_week) or (task["day"] > current_day)) and not("completed", False):

                            progress["current_month"] = month["month"]
                            progress["current_week"] = week["week"]
                            progress["current_day"] = task["day"]
                            return
    #If no more tasks, mark as completed
    progress["current_day"] = -1  #to indicate completion

# YOUTUBE VIDEO RECOMMENDATION
def get_current_week_videos(user_id: str) -> dict:
    """Get Youtube videos for current week's focus"""

    roadmap = user_store.get(user_id)
    if not roadmap:
        return{"error": "User not found"}
    
    progress = roadmap.get("progress", {})
    current_month = progress.get("current_month", 1)
    current_week = progress.get("current_week", 1)

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
    
    videos = search_youtube_videos(current_week_focus)

    return {
        "week_focus": current_week_focus,
        "week_info": f"Month {current_month}, Week {current_week}",
        "videos": videos,
        "total_videos": len(videos)
    }

def search_youtube_videos(query: str, req:FullPipelineReq,max_results: int = 8) -> list:
    """Search Youtube for target role videos"""

    if not YOUTUBE_API_KEY:
        print("No Youtube API key - returning sample videos")
        return get_sample_videos(query)
    
    try:
        enhanced_query = f"{query} {req:target_role} tutorial coding"

        with httpx.Client(timeout=30.0) as client:
            #search videos
            search_response = client.get(
                "https://www.goggleapis.com/youtube/v3/search",params={
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
def get_ai_chat_response(user_id: str, message: str, req:FullPipelineReq) -> dict:
    """Generate AI chat response based on user's roadmap context"""

    roadmap = user_store.get(user_id)
    if not roadmap:
        return {"error": "User not found"}
    
    # Get user context
    user_goal = roadmap.get("goal", "career goal")
    current_progress = roadmap.get("progress", {})
    total_completed = current_progress.get("total_task_completed", 0)

    # Get chat history
    if user_id not in chat_history:
        chat_history[user_id] = []

    #Build AIs context-aware prompt
    system_prompt = f"""You are Navi, a helpful career mentor AI assitant

Context about the user:
- Career Goal: {user_goal}
- Tasks Completed: {total_completed}
- Learning Journey: Currently working on their {req.target_role} roadmap

Your role:
1. Answer questions about {req.target_role}, career development, and learning
2. Provide encouragement and motivation
3. Give practical advice based on their goal
4. Keep responses conversational and supportive
5. If asked about progress, reference their completed tasks

Guidelines:
- Be encouraging and positive
- Provide practical, actionable advice
- Keep responses under 200 words unless more detail is needed
- Reference their goal when relevant
"""
    
    # Prepare conversation history
    messages = [{"role": "system", "content": system_prompt}]

    # Add recent chat history (last  6 messages to stay within token limits)
    recent_history = chat_history[user_id][-6:]
    for chat in recent_history:
        messages.append({"role": "user", "content": chat["user"]})
        message.append({"role": "assistant", "content": chat["assistant"]})

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
                    "model": "deepseek-r1-distill-llama-70b",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            )
        response.raise_for_status()
        ai_response = response.json()["choices"][0]["message"]["content"]

        # Store in chat history 
        chat_history[user_id].append({
            "user": message,
            "assistant": ai_response,
            "timestamp": datetime.now().isoformat()
        })

        #Let's keep only the last 20 conversations to manage memory
        if len(chat_history[user_id]) > 20:
            chat_history[user_id] = chat_history[user_id][-20:]
        
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

@app.post("api/generate_roadmap")
def api_generate_roadmap(req: FullPipelineReq):
    """Generate intial roadmap"""
    try:
        roadmap = llm_generate_roadmap(req)
        user_id = str(uuid4())
        user_store[user_id] = roadmap

        return {
            "success": True,
            "user_id": user_id,
            "roadmap": roadmap,
            "message": "Roadmap generated successfully!"
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to generate roadmap: {str(e)}")
    
@app.get("/api/daily_task/{user_id}")
def api_get_daily_task(user_id:str):
    """Get current daily task with motivation"""
    if user_id not in user_store:
        raise HTTPException(404, "User not found")
    
    task = get_current_daily_task(user_id)
    if not task:
        raise HTTPException(404, "No current task found")
    
    return task

@app.post("/api/complete_task/{user_id}")
def api_complete_task(user_id:str, completion: TaskCompletion):
    """Mark current task as completed"""
    if user_id not in user_store:
        raise HTTPException(404, "user not found")
    
    # Get current task ID

    current_task = get_current_daily_task(user_id)
    if not current_task or "task_id" not in current_task:
        raise HTTPException(404, "No current task to complete")
    
    result = mark_task_completed(user_id, current_task["task_id"])
    
    if "error" in result: 
        raise HTTPException(400, result["error"])
    
    return result


@app.get("/api/week_videos/{user_id}")
def api_get_week_videos(user_id: str):
    """Get Youtube videos for current week"""
    if user_id not in user_store:
        raise HTTPException(404, "user not found")
    
    videos = get_current_week_videos(user_id)

    if "error" in videos:
        raise HTTPException(404, videos["error"])
    
    return videos

@app.post("/api/chat/{user_id}")
async def api_chat(user_id: str, chat_msg: ChatMessage):
    """Chat with AI assistant"""
    if user_id not in user_store:
        raise HTTPException(404, "User not found")
    
    response = get_ai_chat_response(user_id, chat_msg.message, None)  # Added None as req parameter
    if "error" in response:
        raise HTTPException(400, response["error"])
    
    return response

@app.get("/api/user_progress/{user_id}")
def api_get_user_progress(user_id: str):
    """Get user's overall progress"""
    if user_id not in user_store:
        raise HTTPException(404, "User not found")
    
    roadmap = user_store[user_id]
    progress = roadmap.get("progress", {})

    # Calculate completion percentage
    total_tasks = 0
    completed_tasks = progress.get("total_tasks_completed", 0)

    for month in roadmap.get("roadmap", []):
        for week in month["weeks"]:
            total_tasks += len(week.get("daily_tasks", []))
    
    completion_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0

    return {
        "goal": roadmap.get("goal", ""),
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "completion_percentage": round(completion_percentage, 1),
        "current_month": progress.get("current_month", 1),
        "current_week": progress.get("current_week", 1),
        "current_day": progress.get("current_day", 1),
        "start_date": progress.get("start_date", 1)
    }


@app.get("api/health")
def health_check():
    return {
        "status": "healthy",
        "active_users": len(user_store),
        "grow_configured": bool(GROQ_API_KEY),
        "youtube_configured": bool(YOUTUBE_API_KEY)
    }


#  Legacy endpoint for compatibility
@app.post("/api/full_pipeline")
def api_full_pipeline(req: FullPipelineReq):
    """Legacy endpoint - redirects to new generate_roadmap"""
    return api_generate_roadmap(req)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent_orhcestra:app", host="127.0.0.1", port=8000, reload=True)