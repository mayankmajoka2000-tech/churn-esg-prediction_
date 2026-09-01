const express = require("express");
const path = require("path");
const fetch = require("node-fetch");

const app = express();

const PORT = process.env.PORT || 3000;

const API_BASE_URL =
  process.env.API_BASE_URL ||
  "http://localhost:8000";


/* =========================
   EXPRESS CONFIGURATION
========================= */

app.set("view engine", "ejs");

app.set(
  "views",
  path.join(__dirname, "views")
);


/* =========================
   MIDDLEWARE
========================= */

app.use(
  express.urlencoded({
    extended: true
  })
);

app.use(express.json());

app.use(
  express.static(
    path.join(__dirname, "public")
  )
);


/* =========================
   DEFAULT CUSTOMER VALUES
========================= */

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


/* =========================
   HOME PAGE
========================= */

app.get("/", (req, res) => {

  res.render("index", {

    values: defaults,

    error: null,

    apiBaseUrl: API_BASE_URL

  });

});


/* =========================
   PREDICTION
========================= */

app.post("/predict", async (req, res) => {

  const values = {
    ...defaults,
    ...req.body
  };


  /* =========================
     PREPARE API PAYLOAD
  ========================== */

  const payload = {

    gender: values.gender,

    SeniorCitizen:
      Number(values.SeniorCitizen),

    Partner: values.Partner,

    Dependents:
      values.Dependents,

    tenure:
      Number(values.tenure),

    Contract:
      values.Contract,

    PaperlessBilling:
      values.PaperlessBilling,

    PaymentMethod:
      values.PaymentMethod,

    MonthlyCharges:
      Number(values.MonthlyCharges),

    TotalCharges:
      Number(values.TotalCharges),

    PhoneService:
      values.PhoneService,

    MultipleLines:
      values.MultipleLines,

    InternetService:
      values.InternetService,

    OnlineSecurity:
      values.OnlineSecurity,

    OnlineBackup:
      values.OnlineBackup,

    DeviceProtection:
      values.DeviceProtection,

    TechSupport:
      values.TechSupport,

    StreamingTV:
      values.StreamingTV,

    StreamingMovies:
      values.StreamingMovies

  };


  try {

    /* =========================
       CALL FASTAPI
    ========================== */

    const response = await fetch(
      `${API_BASE_URL}/predict`,
      {
        method: "POST",

        headers: {
          "Content-Type":
            "application/json"
        },

        body: JSON.stringify(payload)

      }
    );


    /* =========================
       READ RESPONSE
    ========================== */

    const data =
      await response.json();


    /* =========================
       HANDLE API ERROR
    ========================== */

    if (!response.ok) {

      const message =
        Array.isArray(data.detail)

          ? data.detail
              .map(x => x.msg)
              .join("; ")

          : (
              data.detail ||
              "Prediction failed."
            );


      return res.render(
        "index",
        {

          values,

          error: message,

          apiBaseUrl:
            API_BASE_URL

        }
      );

    }


    /* =========================
       SUCCESS
    ========================== */

    res.render(
      "result",
      {

        result: data,

        customer: values,

        apiBaseUrl:
          API_BASE_URL

      }
    );


  } catch (err) {

    console.error(
      "Prediction API error:",
      err
    );


    res.render(
      "index",
      {

        values,

        error:
          `Unable to reach the prediction API at ${API_BASE_URL}. Please check that the FastAPI backend is running.`,

        apiBaseUrl:
          API_BASE_URL

      }
    );

  }

});


/* =========================
   START SERVER
========================= */

app.listen(
  PORT,
  "0.0.0.0",
  () => {

    console.log(
      `Churn ESG frontend running on port ${PORT}`
    );

    console.log(
      `FastAPI backend configured at ${API_BASE_URL}`
    );

  }
);
