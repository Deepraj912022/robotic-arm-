const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json());

// ================= CONFIG =================
const HTTP_PORT = process.env.PORT || 3000;
const MONGO_URI =  "mongodb+srv://Maitreya:killdill12@cluster0.sk6ugig.mongodb.net/myDatabase?retryWrites=true&w=majority";

// ================= MONGODB =================
mongoose.connect(MONGO_URI)
.then(() => console.log("✅ MongoDB connected"))
.catch(err => console.error("❌ Mongo error:", err));

// ================= SCHEMA =================
const coordSchema = new mongoose.Schema({
  mode: String,
  timestamp: Number,
  coords: Array
}, { timestamps: true });

const Coord = mongoose.model("Coord", coordSchema);

// ================= API =================

// 🔥 IMPORTANT: Python sends data here
app.post("/coords", async (req, res) => {
  try {
    const data = req.body;

    if (!data || !data.coords) {
      return res.status(400).json({ error: "Invalid payload" });
    }

    // Keep only latest
    await Coord.deleteMany({});
    await Coord.create(data);

    res.json({ status: "saved" });

  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// 🔹 Godot fetches from here
app.get("/coords", async (req, res) => {
  try {
    const latest = await Coord.findOne().sort({ createdAt: -1 });

    if (!latest) {
      return res.json({ message: "No data yet" });
    }

    res.json(latest);

  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// 🔹 Health
app.get("/", (req, res) => {
  res.send("✅ Backend running");
});

// ================= START =================
app.listen(HTTP_PORT, () => {
  console.log(`🌐 Server running on port ${HTTP_PORT}`);
});
