# KMRL Alert Detection Keywords
# Critical keywords that indicate severity levels and departments

SEVERITY_KEYWORDS = {
    "critical": {
        "keywords": [
            "emergency", "derailment", "collision", "fire", "explosion", 
            "evacuation", "accident", "death", "injury", "fatal", "critical", "danger", 
            "immediate", "urgent", "stop", "halt", "shutdown", "alarm", 
            "destroyed", "severe", "catastrophic"
        ],
        "weight": 4.0,
        "threshold": 1  # minimum keywords needed
    },
    "high": {
        "keywords": [
            "failure", "fault", "defect", "problem", "issue", "concern", "abnormal", "irregular", 
            "unexpected", "unusual", "suspicious", "error", "malfunction",
            "delay", "disruption", "interrupt", "block", "obstruct", "signal", 
            "power", "electrical", "mechanical", "hydraulic", "pneumatic", "brake",
            "engine", "motor", "wheel", "track", "rail", "overhead", "contact", "breakdown"
        ],
        "weight": 3.0,
        "threshold": 1
    },
    "medium": {
        "keywords": [
            "maintenance", "repair", "service", "check", "inspect", "replace", "adjust",
            "clean", "lubricate", "calibrate", "test", "monitor", "observe", "report",
            "schedule", "routine", "periodic", "preventive", "corrective", "minor",
            "wear", "noise", "vibration", "temperature", "pressure", "voltage", "warning"
        ],
        "weight": 2.0,
        "threshold": 1
    },
    "low": {
        "keywords": [
            "information", "notice", "update", "status", "normal", "operational", 
            "running", "working", "functional", "routine", "standard", "regular",
            "completed", "finished", "done", "ok", "good", "fine", "stable",
            "passenger", "announcement", "schedule", "time", "arrival", "departure",
            "minor"
        ],
        "weight": 1.0,
        "threshold": 1
    }
}

DEPARTMENT_KEYWORDS = {
    "operations": {
        "keywords": [
            "train", "service", "schedule", "passenger", "station", "platform", "ticket",
            "conductor", "driver", "operator", "control", "dispatch", "timetable",
            "departure", "arrival", "delay", "cancel", "route", "line", "service",
            "announcement", "boarding", "safety", "emergency", "evacuation"
        ],
        "weight": 3.0,
        "threshold": 1
    },
    "maintenance": {
        "keywords": [
            "repair", "fix", "replace", "service", "maintain", "check", "inspect",
            "clean", "lubricate", "adjust", "calibrate", "overhaul", "upgrade",
            "component", "part", "system", "equipment", "machinery", "tool",
            "spare", "inventory", "workshop", "depot", "facility", "technical"
        ],
        "weight": 3.0,
        "threshold": 1
    },
    "safety": {
        "keywords": [
            "safety", "security", "protection", "risk", "hazard", "danger", "warning",
            "precaution", "protocol", "procedure", "guideline", "regulation", "compliance",
            "accident", "incident", "injury", "damage", "fire", "emergency", "evacuation",
            "first", "aid", "medical", "ambulance", "police", "investigation"
        ],
        "weight": 3.0,
        "threshold": 1
    }
}

# Context keywords that boost relevance
CONTEXT_KEYWORDS = {
    "railway": [
        "kmrl", "kochi", "metro", "rail", "railway", "train", "coach", "locomotive",
        "track", "signal", "station", "platform", "depot", "workshop", "control",
        "overhead", "contact", "wire", "power", "electrical", "mechanical"
    ],
    "location": [
        "aluva", "kalamassery", "edapally", "changampuzha", "palarivattom", 
        "kaloor", "lissie", "ernakulam", "maharaja", "mg", "road", "ernakulam",
        "town", "vytilla", "thaikoodam", "petta", "kadavanthra", "elamkulam"
    ],
    "time": [
        "morning", "afternoon", "evening", "night", "peak", "hour", "rush",
        "schedule", "timetable", "delay", "early", "late", "on", "time"
    ]
}

def get_keyword_score(text, keyword_dict):
    """Calculate keyword-based score for text"""
    text_lower = text.lower()
    scores = {}
    
    for category, data in keyword_dict.items():
        keywords = data["keywords"]
        weight = data["weight"]
        threshold = data.get("threshold", 1)
        
        # Count keyword matches
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Calculate score
        if matches >= threshold:
            scores[category] = matches * weight
        else:
            scores[category] = 0
    
    return scores

def get_context_boost(text):
    """Calculate context relevance boost"""
    text_lower = text.lower()
    boost = 0
    
    for context_type, keywords in CONTEXT_KEYWORDS.items():
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        if context_type == "railway":
            boost += matches * 0.5
        else:
            boost += matches * 0.2
    
    return boost

def analyze_alert_keywords(text):
    """Main function to analyze text for alert keywords"""
    # Get severity scores
    severity_scores = get_keyword_score(text, SEVERITY_KEYWORDS)
    
    # Get department scores  
    department_scores = get_keyword_score(text, DEPARTMENT_KEYWORDS)
    
    # Get context boost
    context_boost = get_context_boost(text)
    
    # Apply context boost only to categories that have matches
    for severity, score in severity_scores.items():
        if score > 0:  # Only boost if there are actual matches
            severity_scores[severity] += context_boost
        
    for dept, score in department_scores.items():
        if score > 0:  # Only boost if there are actual matches
            department_scores[dept] += context_boost
    
    # Determine predictions
    predicted_severity = max(severity_scores, key=severity_scores.get)
    predicted_department = max(department_scores, key=department_scores.get)
    
    # Calculate confidence (normalized scores)
    total_severity = sum(severity_scores.values())
    total_dept = sum(department_scores.values())
    
    max_severity_score = max(severity_scores.values())
    max_dept_score = max(department_scores.values())
    
    severity_confidence = max_severity_score / total_severity if total_severity > 0 else 0
    dept_confidence = max_dept_score / total_dept if total_dept > 0 else 0
    
    # Handle case where no keywords match at all
    if total_severity == 0:
        predicted_severity = "low"
        severity_confidence = 0.1
    if total_dept == 0:
        predicted_department = "operations" 
        dept_confidence = 0.1
    
    return {
        "severity": predicted_severity,
        "department": predicted_department,
        "severity_scores": severity_scores,
        "department_scores": department_scores,
        "severity_confidence": severity_confidence,
        "department_confidence": dept_confidence,
        "context_boost": context_boost,
        "overall_confidence": (severity_confidence + dept_confidence) / 2
    }