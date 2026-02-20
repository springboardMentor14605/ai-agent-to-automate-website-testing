const apiBaseURL = "http://localhost:5000/api";

export default {
  post: async (endpoint, data) => {
    const response = await fetch(`${apiBaseURL}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });
    if (!response.ok) throw response;
    return response.json();
  }
};