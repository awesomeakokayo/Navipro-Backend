# Navi - AI-Powered Learning Roadmap Generator

A FastAPI-based backend service that generates personalized learning roadmaps using AI, tracks progress, and provides daily learning tasks with motivational support.

## Features

- **AI-Powered Roadmap Generation**: Creates structured learning plans using Groq's LLM
- **Progress Tracking**: Monitors daily task completion and overall progress
- **Personalized Daily Tasks**: Provides tailored daily learning activities
- **YouTube Integration**: Recommends relevant learning videos
- **AI Chat Assistant**: Context-aware career coaching and guidance
- **Multi-User Support**: Secure user authentication and data isolation

## Tech Stack

- **Backend**: FastAPI, Python
- **Database**: SQLAlchemy with SQLite/PostgreSQL support
- **AI/ML**: Groq API for LLM interactions
- **Authentication**: JWT-based user identification
- **External APIs**: YouTube Data API
- **Deployment**: Ready for production with Neon PostgreSQL support

## Quick Start

### Prerequisites

- Python 3.8+
- Groq API key
- YouTube Data API key (optional)

## Installation

1. **Clone and setup environment**:
```bash
git clone <repository-url>
cd navi-backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Environment configuration**:
Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
YOUTUBE_API_KEY=your_youtube_api_key_optional
DATABASE_URL=sqlite:///./navi.db
AUTH_SECRET=your_jwt_secret_here
```

4. **Run the application**:
```bash
python agent_orchestra.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Core Routes

- `POST /api/generate_roadmap` - Create personalized learning roadmap
- `GET /api/user_roadmap` - Retrieve user's roadmap
- `GET /api/daily_task` - Get current daily task
- `POST /api/complete_task` - Mark task as completed
- `GET /api/week_videos` - Get learning videos for current week
- `POST /api/chat` - AI career coaching chat
- `GET /api/user_progress` - Get progress statistics
- `GET /api/health` - Health check

### Authentication

Include either:
- `X-User-ID` header with user identifier, OR
- `Authorization: Bearer <jwt-token>` header

## Roadmap Structure

Generated roadmaps follow this hierarchical structure:
- **Months** (3, 6, or 12 based on timeframe)
  - **Weeks** (4 weeks per month)
    - **Daily Tasks** (6 tasks per week)

Each task includes:
- Title and description
- Estimated time
- Learning resources
- Completion tracking

## Database Models

- **Users**: User profiles and learning preferences
- **Roadmaps**: Generated learning plans in JSON format
- **Progress**: Task completion and current position
- **ChatHistory**: Conversation history with AI assistant

## Configuration

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `GROQ_API_KEY` | Groq API authentication | Required |
| `GROQ_BASE_URL` | Groq API endpoint | `https://api.groq.com/openai/v1` |
| `YOUTUBE_API_KEY` | YouTube Data API key | Optional |
| `DATABASE_URL` | Database connection string | `sqlite:///./navi.db` |
| `AUTH_SECRET` | JWT verification secret | Required |

### Database Support

- **SQLite** (default): Good for development
- **PostgreSQL**: Recommended for production
- **Neon PostgreSQL**: Optimized for serverless deployment

## Deployment

### Local Development
```bash
python agent_orchestra.py
```

### Production with Uvicorn
```bash
uvicorn agent_orchestra:app --host 0.0.0.0 --port $PORT
```

### Docker (Example)
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "agent_orchestra:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Example Usage

### Generate a Roadmap
```bash
curl -X POST "http://localhost:8000/api/generate_roadmap" \
  -H "X-User-ID: user123" \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Become a Full Stack Developer",
    "target_role": "Frontend Developer",
    "timeframe": "6_months",
    "hours_per_week": "15",
    "learning_style": "visual",
    "learning_speed": "average",
    "skill_level": "beginner"
  }'
```

### Get Daily Task
```bash
curl -X GET "http://localhost:8000/api/daily_task" \
  -H "X-User-ID: user123"
```

## AI Features

### Roadmap Generation
- Uses Groq's Qwen model for structured roadmap creation
- Fallback system ensures roadmap availability
- Validates structure and content quality

### Career Coaching
- Context-aware responses based on user progress
- Motivational messaging and progress encouragement
- Career guidance and learning advice

## Error Handling

- Comprehensive retry logic for LLM calls
- Graceful fallbacks for missing API keys
- Structured error responses with debugging information

## Monitoring

- Health check endpoint for service status
- Detailed logging for debugging
- Progress tracking and analytics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

MIT or as specified in the repository.

## Support

For issues and questions:
1. Check the health endpoint
2. Verify API key configuration
3. Review application logs
4. Open a GitHub issue

---

Built with ❤️ for lifelong learners and career changers.
