const express = require("express");
const path = require("path");
const fetch = require("node-fetch");

const app = express();

const PORT = process.env.PORT || 3000;

// Live FastAPI backend on Render
const API_BASE_URL =
  process.env.API_BASE_URL ||
  "https://churn-esg-prediction.onrender.com";

app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));

app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// Default customer values
const defaults = {
  gender: "Female",
  SeniorCitizen: "0",
  Partner: "Yes",
  Dependents: "No",

  tenure: "12",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",

  MonthlyCharges: "70.35",
  TotalCharges: "845.50",

  PhoneService: "Yes",
  MultipleLines: "No",

  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "Yes",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "Yes",
  StreamingMovies: "No"
};


// =========================
// HOME PAGE
// =========================

app.get("/", (req, res) => {
  res.render("index", {
    values: defaults,
    error: null,
    apiBaseUrl: API_BASE_URL
  });
});


// =========================
// PREDICTION
// =========================

app.post("/predict", async (req, res) => {

  // Keep submitted values so the form can be restored
  const values = {
    ...defaults,
    ...req.body
  };

  // Convert form strings into the correct Python/Pydantic types
  const payload = {
    gender: values.gender,

    SeniorCitizen: Number(values.SeniorCitizen),

    Partner: values.Partner,
    Dependents: values.Dependents,

    tenure: Number(values.tenure),

    Contract: values.Contract,

    PaperlessBilling: values.PaperlessBilling,

    PaymentMethod: values.PaymentMethod,

    MonthlyCharges: Number(values.MonthlyCharges),

    TotalCharges: Number(values.TotalCharges),

    PhoneService: values.PhoneService,

    MultipleLines: values.MultipleLines,

    InternetService: values.InternetService,

    OnlineSecurity: values.OnlineSecurity,

    OnlineBackup: values.OnlineBackup,

    DeviceProtection: values.DeviceProtection,

    TechSupport: values.TechSupport,

    StreamingTV: values.StreamingTV,

    StreamingMovies: values.StreamingMovies
  };


  // Basic numeric validation
  if (
    Number.isNaN(payload.SeniorCitizen) ||
    Number.isNaN(payload.tenure) ||
    Number.isNaN(payload.MonthlyCharges) ||
    Number.isNaN(payload.TotalCharges)
  ) {
    return res.render("index", {
      values,
      error: "Please enter valid numeric values for tenure and charges.",
      apiBaseUrl: API_BASE_URL
    });
  }


  try {

    console.log("Sending prediction request to:", API_BASE_URL);
    console.log("Payload:", payload);


    // Call FastAPI
    const response = await fetch(
      `${API_BASE_URL}/predict`,
      {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json"
        },

        body: JSON.stringify(payload)
      }
    );


    // Read API response
    const data = await response.json();


    console.log("FastAPI response:", data);


    // Handle FastAPI validation/server errors
    if (!response.ok) {

      const message = Array.isArray(data.detail)
        ? data.detail
            .map((item) => item.msg)
            .join("; ")
        : (
            data.detail ||
            "Prediction failed."
          );


      return res.render("index", {
        values,
        error: message,
        apiBaseUrl: API_BASE_URL
      });
    }


    // Successful prediction
    return res.render("result", {
      result: data,
      customer: values
    });


  } catch (err) {

    console.error("Prediction API error:", err);


    return res.render("index", {
      values,

      error:
        `Unable to reach the prediction API. ` +
        `Backend: ${API_BASE_URL}. ` +
        `Please check that the FastAPI Render service is running.`,

      apiBaseUrl: API_BASE_URL
    });
  }
});


// =========================
// ERROR HANDLER
// =========================

app.use((err, req, res, next) => {

  console.error("Unexpected server error:", err);

  res.status(500).send(
    "Internal server error. Please check the frontend logs."
  );
});


// =========================
// START SERVER
// =========================

app.listen(PORT, () => {

  console.log(
    `Churn ESG frontend running on port ${PORT}`
  );

  console.log(
    `FastAPI backend configured at: ${API_BASE_URL}`
  );

});
