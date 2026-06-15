import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json'
  }
})

export const sendMessage = async (sessionId, userInput, agentState) => {
  const response = await api.post('/chat', {
    session_id: sessionId,
    user_input: userInput,
    agent_state: agentState
  })
  return response.data
}

export const uploadFile = async (sessionId, file, fileType) => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('session_id', sessionId)
  formData.append('file_type', fileType)
  
  const response = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return response.data
}

export const analyzeJob = async (sessionId, jobDescription, agentState) => {
  const response = await api.post('/analyze-job', {
    session_id: sessionId,
    job_description: jobDescription,
    agent_state: agentState
  })
  return response.data
}

export const reviewDocument = async (sessionId, documentType, agentState) => {
  const response = await api.post('/review-document', {
    session_id: sessionId,
    document_type: documentType,
    agent_state: agentState
  })
  return response.data
}

export const deleteDocument = async (sessionId, documentType) => {
  const response = await api.post('/delete-document', {
    session_id: sessionId,
    document_type: documentType
  })
  return response.data
}