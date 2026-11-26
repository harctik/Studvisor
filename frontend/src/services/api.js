import axios from "axios";
import { getToken, removeToken } from "../utils/auth";
import useAuthStore from "../stores/authStore";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 20000,
});

api.interceptors.request.use((config) => {
  const token = getToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

function endSession() {
  removeToken();
  useAuthStore.getState().logout();
  window.location.assign("/login");
}

api.interceptors.response.use(undefined, (error) => {
  if (error.response?.status === 401 && !error.config?.url?.endsWith("/login")) {
    endSession();
  }
  return Promise.reject(error);
});

/** Turn a failed request into a sentence a student can act on. */
export function errorMessage(error) {
  const response = error?.response;
  if (!response) {
    return `Can't reach the Studvisor server at ${API_BASE_URL}.`;
  }
  if (response.status >= 500) {
    return "The server hit an error. Please try again.";
  }
  const text = response.data?.message || response.data?.detail;
  return typeof text === "string" ? text : `Request failed (${response.status}).`;
}

const get = (url, params) => api.get(url, { params }).then((r) => r.data);
const post = (url, body, params) => api.post(url, body, { params }).then((r) => r.data);

// Collection routes on the backend end in a slash.

// Account
export const loginUser = (credentials) => post("/login", credentials);
export const fetchProfile = () => get("/student/me");

// Attendance
export const fetchOverallAttendance = () => get("/attendance/overall");
export const fetchSubjectAttendance = () => get("/attendance/subject-wise");
export const fetchMissedClasses = () => get("/attendance/missed-classes");
export const fetchBunkAlerts = () => get("/attendance/bunk-alerts");
export const fetchHeatmap = () => get("/attendance/heatmap");
export const simulateBunk = (missCount = 1, subjectId = null) =>
  get("/attendance/simulate-bunk", {
    miss_count: missCount,
    ...(subjectId ? { subject_id: subjectId } : {}),
  });

// Academics
export const fetchMarks = () => get("/marks/");
export const fetchCGPA = () => get("/gpa/cgpa");
export const predictCgpa = (targetPct) => get("/analytics/predict-cgpa", { target_pct: targetPct });
export const fetchTimetable = () => get("/timetable/");
export const fetchCalendar = () => get("/calendar/");
export const fetchAssignments = () => get("/assignments/");
export const submitAssignment = (id) => post(`/assignments/${id}/submit`);
export const fetchExams = () => get("/exams/");
export const fetchSyllabus = () => get("/syllabus/");
export const fetchNotes = () => get("/notes/");
export const rateNote = (id, helpful) => post(`/notes/${id}/rate`, null, { helpful });

// Leave
export const fetchLeaveRequests = () => get("/leave/requests");
export const fetchLeaveBalance = () => get("/leave/balance");
export const applyLeaveRequest = (data) => post("/leave/requests", data);
export const withdrawLeaveRequest = (id) => api.delete(`/leave/requests/${id}`).then((r) => r.data);

// Fees
export const fetchMyFees = () => get("/fees/my-fees");
export const fetchFeeSummary = () => get("/fees/summary");
export const fetchPaymentHistory = () => get("/fees/payments");

// Library and placement
export const searchLibrary = (q) => get("/library/catalog", q ? { q } : {});
export const fetchMyBooks = () => get("/library/my-books");
export const issueBook = (id) => post(`/library/issue/${id}`);
export const returnBook = (issueId) => post(`/library/return/${issueId}`);
export const fetchDrives = () => get("/placement/drives");
export const fetchApplications = () => get("/placement/my-applications");
export const applyToDrive = (id) => post(`/placement/apply/${id}`);

// Campus
export const fetchAnnouncements = () => get("/announcements/");
export const fetchNotifications = () => get("/notifications/");
export const fetchEvents = () => get("/events/");
export const rsvpEvent = (id) => post(`/events/${id}/rsvp`);
export const fetchPolls = () => get("/polls/");
export const votePoll = (pollId, optionId) => post(`/polls/${pollId}/vote`, null, { option_id: optionId });
export const fetchFaculty = () => get("/faculty/directory");
export const fetchLostFound = () => get("/lost-found/");
export const createLostFoundItem = (data) => post("/lost-found/", data);
export const fetchAnonPosts = () => get("/anon/posts");
export const createAnonPost = (category, content) => post("/anon/posts", { category, content });
export const fetchComplaints = () => get("/complaints/");
export const createComplaint = (data) => post("/complaints/", data);
export const fetchMerit = () => get("/merit/my");
export const fetchAchievements = () => get("/achievements/my");
export const fetchLeaderboard = () => get("/leaderboard/");

// Assistant
export const fetchChatHistory = () => get("/chat/history");
export const clearChatHistory = () => api.delete("/chat/history").then((r) => r.data);
export const fetchChatSuggestions = () => get("/chat/suggestions");
export const sendChatMessage = (query, contextPage = "dashboard") =>
  post("/chat/", { query, context_page: contextPage });

/** POST /chat/stream answers with `data: {"token": "...", "done": false}` events. */
export async function streamChatMessage(query, { contextPage = "dashboard", onText } = {}) {
  const token = getToken();
  const response = await fetch(`${API_BASE_URL}/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ query, context_page: contextPage }),
  });

  if (response.status === 401) endSession();
  if (!response.ok || !response.body) {
    throw new Error("The assistant didn't answer.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let text = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split(/\r?\n\r?\n/);
    buffer = events.pop() || "";

    for (const event of events) {
      if (!event.startsWith("data:")) continue;
      text += JSON.parse(event.slice(5)).token ?? "";
      onText?.(text);
    }
  }

  return text;
}

// ── ADMIN PANEL ─────────────────────────────────────────────────────────────

export async function fetchAdminDashboard() {
  const response = await api.get("/admin/dashboard");
  return response.data;
}

export async function fetchAllStudents() {
  const response = await api.get("/admin/students");
  return response.data;
}

export async function fetchAllFaculty() {
  const response = await api.get("/admin/faculty");
  return response.data;
}

export async function fetchPendingLeaves() {
  const response = await api.get("/admin/leaves/pending");
  return response.data;
}

export async function updateLeaveStatus(leaveId, status) {
  const response = await api.put(`/admin/leaves/${leaveId}`, { status });
  return response.data;
}

export default api;

