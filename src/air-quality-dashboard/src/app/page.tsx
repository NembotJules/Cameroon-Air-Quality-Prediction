"use client"

import React, { useState } from 'react'
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet'
import { Line, LineChart, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Flag, Info } from 'lucide-react'

// Placeholder data for Cameroon cities
const cities = [
  { name: "Douala", lat: 4.0511, lon: 9.7679 },
  { name: "Yaoundé", lat: 3.8480, lon: 11.5021 },
  { name: "Buéa", lat: 4.1537, lon: 9.2920 },
  { name: "Bafoussam", lat: 5.4768, lon: 10.4214 },
  { name: "Ngaoundéré", lat: 7.3220, lon: 13.5843 },
  { name: "Bamenda", lat: 5.9597, lon: 10.1417 },
  { name: "Garoua", lat: 9.3017, lon: 13.3921 },
  { name: "Maroua", lat: 10.5910, lon: 14.3158 },
  { name: "Kumba", lat: 4.6363, lon: 9.4464 },
  { name: "Nkongsamba", lat: 4.9547, lon: 9.9404 },
]

// Function to generate random air quality data
const generateAirQualityData = () => {
  return cities.map(city => ({
    ...city,
    pm25: Math.floor(Math.random() * 150) + 1 // Random PM2.5 value between 1-150
  }))
}

// Function to get color based on PM2.5 value
const getColor = (pm25: number) => {
  if (pm25 <= 50) return "green"
  if (pm25 <= 100) return "yellow"
  if (pm25 <= 150) return "orange"
  return "red"
}

// Generate mock forecast data
const generateForecastData = (city: string) => {
  return Array.from({ length: 7 }, (_, i) => ({
    date: new Date(Date.now() + i * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    pm25: Math.floor(Math.random() * 150) + 1
  }))
}

export default function AirQualityDashboard() {
  const [selectedDate, setSelectedDate] = useState("Today")
  const [selectedCity, setSelectedCity] = useState("Douala")
  const [airQualityData, setAirQualityData] = useState(generateAirQualityData())
  const [forecastData, setForecastData] = useState(generateForecastData(selectedCity))

  const handleDateChange = (date: string) => {
    setSelectedDate(date)
    setAirQualityData(generateAirQualityData()) // Simulate new data for different dates
  }

  const handleCityChange = (city: string) => {
    setSelectedCity(city)
    setForecastData(generateForecastData(city))
  }

  return (
    <div className="container mx-auto p-4">
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="text-2xl font-bold flex items-center">
            PM2.5 Predictions for Cameroon
            <Flag className="ml-2" />
          </CardTitle>
          <CardDescription>Select forecasting day</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="mb-4 flex space-x-2">
            {["Today", "Tomorrow", "2023-10-27", "2023-10-28", "2023-10-29"].map((date) => (
              <Button
                key={date}
                onClick={() => handleDateChange(date)}
                variant={selectedDate === date ? "default" : "outline"}
              >
                {date}
              </Button>
            ))}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="h-[400px] w-full">
              <MapContainer center={[7.3697, 12.3547]} zoom={6} style={{ height: "100%", width: "100%" }}>
                <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                {airQualityData.map((city, index) => (
                  <Marker key={index} position={[city.lat, city.lon]}>
                    <Popup>
                      <div>
                        <h3 className="font-bold">{city.name}</h3>
                        <p>PM2.5: {city.pm25}</p>
                        <div
                          className="w-4 h-4 rounded-full inline-block mr-2"
                          style={{ backgroundColor: getColor(city.pm25) }}
                        ></div>
                        {getColor(city.pm25).charAt(0).toUpperCase() + getColor(city.pm25).slice(1)}
                      </div>
                    </Popup>
                  </Marker>
                ))}
              </MapContainer>
            </div>
            <div>
              <div className="mb-4">
                <Select onValueChange={handleCityChange} defaultValue={selectedCity}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a city" />
                  </SelectTrigger>
                  <SelectContent>
                    {cities.map((city) => (
                      <SelectItem key={city.name} value={city.name}>
                        {city.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={forecastData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="pm25" stroke="#8884d8" name="PM2.5" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <Card className="mt-4">
            <CardHeader>
              <CardTitle className="text-lg font-semibold flex items-center">
                <Info className="mr-2" />
                How it's done?
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p>
                Here, we use meta data about the cities (such as population density) together with weather forecasting data to predict the
                magnitude of PM2.5. The motivation lies behind the fact that during certain weather conditions, air quality in Cameroon can be
                significantly affected. Factors such as temperature, humidity, and wind patterns play crucial roles in the concentration of
                particulate matter.
              </p>
            </CardContent>
          </Card>
        </CardContent>
      </Card>
    </div>
  )
}