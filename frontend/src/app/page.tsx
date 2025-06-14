"use client"

import { useState, useEffect, useRef } from "react"
import { Search, Filter, Eye, Users, Wind, LampDeskIcon as Desk } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

const BACKEND_URL = "http://localhost:8000";

function generateRandomFeatures() {
  const allFeatures = [
    "sea view",
    "city view",
    "balcony",
    "air conditioning",
    "desk",
    "double bed",
    "triple bed",
    "max 4 people",
    "max 2 people",
    "max 3 people",
  ]
  const numFeatures = Math.floor(Math.random() * 4) + 2
  return allFeatures.sort(() => 0.5 - Math.random()).slice(0, numFeatures)
}

const predefinedQueries = [
  {
    id: 1,
    title: "Double rooms with a sea view",
    icon: <Eye className="w-4 h-4" />,
    keywords: ["double bed", "sea view"],
  },
  {
    id: 2,
    title: "Rooms with a balcony and air conditioning, with a city view",
    icon: <Wind className="w-4 h-4" />,
    keywords: ["balcony", "air conditioning", "city view"],
  },
  {
    id: 3,
    title: "Triple rooms with a desk",
    icon: <Desk className="w-4 h-4" />,
    keywords: ["triple bed", "desk"],
  },
  {
    id: 4,
    title: "Rooms with a maximum capacity of 4 people",
    icon: <Users className="w-4 h-4" />,
    keywords: ["max 4 people"],
  },
]

export default function ObiletCaseStudy() {
  const [hotelImages, setHotelImages] = useState<any[]>([])
  const [selectedQuery, setSelectedQuery] = useState<number | null>(null)
  const [searchTerm, setSearchTerm] = useState("")
  const [filteredImages, setFilteredImages] = useState<any[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [activeTab, setActiveTab] = useState("predefined")
  const tabRefs = [useRef<HTMLButtonElement>(null), useRef<HTMLButtonElement>(null)]
  const [indicatorStyle, setIndicatorStyle] = useState({ left: 0, width: 0 })

  useEffect(() => {
    // Fetch all images from backend when component mounts
    const fetchAllImages = async () => {
      setIsSearching(true);
      try {
        const res = await fetch(`${BACKEND_URL}/search`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: "" }),
        });
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        const data = await res.json();
        const allImages = data.results || [];
        setHotelImages(allImages);
        setFilteredImages(allImages);
        console.log(`Loaded ${allImages.length} total images`); // Debug log
      } catch (e) {
        console.error("Error fetching images:", e);
        setHotelImages([]);
        setFilteredImages([]);
      }
      setIsSearching(false);
    };
    fetchAllImages();
  }, []);

  useEffect(() => {
    // Animate sliding bar under active tab
    const idx = activeTab === "predefined" ? 0 : 1
    const ref = tabRefs[idx].current
    if (ref) {
      setIndicatorStyle({ left: ref.offsetLeft, width: ref.offsetWidth })
    }
  }, [activeTab, tabRefs[0].current, tabRefs[1].current])

  const handleTabChange = (tab: string) => {
    setActiveTab(tab)
    if (tab === "custom") {
      setSelectedQuery(null)
      setFilteredImages(hotelImages)
    }
  }

  const handleQuerySelect = async (queryId: number) => {
    console.log("Button clicked - Query ID:", queryId); // Debug log for click event

    if (isSearching) {
      console.log("Search already in progress, ignoring click"); // Debug log for search state
      return;
    }
    
    setSelectedQuery(queryId);
    setIsSearching(true);
    setActiveTab("predefined");
    
    const query = predefinedQueries.find((q) => q.id === queryId);
    console.log("Selected query:", query); // Debug log for selected query

    if (query) {
      try {
        // Use the full query title instead of just keywords
        const searchQuery = query.title.toLowerCase();
        console.log("Sending search request to backend:", searchQuery); // Debug log for search query

        const res = await fetch(`${BACKEND_URL}/search`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: searchQuery }),
        });
        
        console.log("Backend response status:", res.status); // Debug log for response status
        
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        
        const data = await res.json();
        console.log("Search results:", data); // Debug log for search results
        
        if (Array.isArray(data.results)) {
          setFilteredImages(data.results);
          console.log(`Found ${data.results.length} matching images`); // Debug log for results count
        } else {
          console.error("Invalid results format:", data); // Debug log for invalid results
          setFilteredImages([]);
        }
      } catch (e) {
        console.error("Search error:", e);
        setFilteredImages([]);
      }
    }
    setIsSearching(false);
  };

  const handleCustomSearch = async () => {
    if (!searchTerm.trim()) {
      setFilteredImages(hotelImages)
      return
    }
    setIsSearching(true)
    try {
      const res = await fetch(`${BACKEND_URL}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: searchTerm }),
      })
      const data = await res.json()
      setFilteredImages(data.results)
    } catch (e) {
      setFilteredImages([])
    }
    setIsSearching(false)
  }

  const resetFilters = () => {
    setSelectedQuery(null)
    setSearchTerm("")
    setFilteredImages(hotelImages)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-red-600 text-white">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-8">
              <h1 className="text-2xl font-bold">oBilet</h1>
              <nav className="hidden md:flex space-x-6">
                <span className="text-red-200">Otobüs</span>
                <span className="text-red-200">Uçak</span>
                <span className="bg-red-700 px-3 py-1 rounded font-medium">Otel</span>
                <span className="text-red-200">Araç</span>
                <span className="text-red-200">Feribot</span>
              </nav>
            </div>
            <div className="flex items-center space-x-4 text-sm">
              <span>TRY</span>
              <span>Yardım</span>
              <span>Rezervasyon Sorgula</span>
              <span>Üye Girişi</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8">
        {/* Case Study Header */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Visual Hotel Room Search - Case Study</h2>
          <p className="text-gray-600 mb-6">
            Advanced hotel room filtering based on visual preferences using AI-powered image analysis.
          </p>

          {/* Search Interface */}
          <div className="relative w-full mb-6">
            <div className="flex w-full bg-gray-100 rounded overflow-hidden relative">
              <button
                ref={tabRefs[0]}
                className={`flex-1 py-2 text-center font-semibold transition-colors z-10 ${activeTab === "predefined" ? "text-red-600" : "text-gray-500"} cursor-pointer`}
                onClick={() => handleTabChange("predefined")}
              >
                Predefined Queries
              </button>
              <button
                ref={tabRefs[1]}
                className={`flex-1 py-2 text-center font-semibold transition-colors z-10 ${activeTab === "custom" ? "text-red-600" : "text-gray-500"} cursor-pointer`}
                onClick={() => handleTabChange("custom")}
              >
                Custom Search
              </button>
              <div
                className="absolute bottom-0 h-1 bg-red-600 rounded transition-all duration-300"
                style={{ left: indicatorStyle.left, width: indicatorStyle.width }}
              />
            </div>
          </div>

          {activeTab === "predefined" ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {predefinedQueries.map((query) => (
                <button
                  key={query.id}
                  className={`w-full text-left focus:outline-none`}
                  onClick={() => {
                    console.log("Raw button click - Query ID:", query.id); // Debug log for raw click
                    handleQuerySelect(query.id);
                  }}
                >
                  <Card
                    className={`cursor-pointer transition-all hover:shadow-md ${
                      selectedQuery === query.id
                        ? "ring-2 ring-red-500 bg-red-50 border-2 border-red-400 shadow-lg"
                        : "border border-gray-200"
                    }`}
                  >
                    <CardContent className="p-4">
                      <div className="flex items-center space-x-3">
                        <div className="text-red-600">{query.icon}</div>
                        <div>
                          <h3 className="font-medium text-gray-900">{query.title}</h3>
                          <div className="flex flex-wrap gap-1 mt-2">
                            {query.keywords.map((keyword, idx) => (
                              <Badge key={idx} variant="secondary" className="text-xs">
                                {keyword}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </button>
              ))}
            </div>
          ) : (
            <div className="flex space-x-2 mb-4">
              <Input
                placeholder="Enter room features (e.g., balcony, sea view, desk...)"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleCustomSearch()}
              />
              <Button onClick={handleCustomSearch} className="bg-red-600 hover:bg-red-700">
                <Search className="w-4 h-4 mr-2" />
                Search
              </Button>
            </div>
          )}

          <div className="flex justify-between items-center mt-6">
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-900">
                {filteredImages.length} of {hotelImages.length} rooms available
              </span>
              {selectedQuery && (
                <Badge variant="outline" className="text-red-600 border-red-600">
                  Query {selectedQuery} Active
                </Badge>
              )}
            </div>
          </div>
        </div>

        {/* Results Section */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-6">Search Results</h3>

          {isSearching ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-red-600"></div>
              <span className="ml-3 text-gray-900">Searching for rooms...</span>
            </div>
          ) : filteredImages.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {filteredImages.map((image) => (
                <Card key={image.url} className="overflow-hidden hover:shadow-lg transition-shadow">
                  <div className="aspect-video bg-gray-200 relative">
                    <img
                      src={image.url || "/placeholder.svg"}
                      alt={image.caption ? image.caption.slice(0, 30) : "Hotel Room"}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        const target = e.target as HTMLImageElement;
                        target.src = `/placeholder.svg?height=200&width=300&text=Room`;
                      }}
                    />
                  </div>
                  <CardContent className="p-4">
                    <h4 className="font-medium text-gray-900 mb-2">
                      Room {image.url ? image.url.split('/').pop().split('.')[0].padStart(2, '0') : '??'}
                    </h4>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <div className="text-gray-400 mb-4">
                <Search className="w-12 h-12 mx-auto" />
              </div>
              <h4 className="text-lg font-medium text-gray-900 mb-2">No rooms found</h4>
              <p className="text-gray-900">Try adjusting your search criteria or selecting a different query.</p>
            </div>
          )}
        </div>

        {/* Technical Implementation Info */}
        <div className="bg-white rounded-lg shadow-sm p-6 mt-8">
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Technical Implementation</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <Eye className="w-8 h-8 text-red-600 mx-auto mb-2" />
              <h4 className="font-medium text-gray-900">Image-to-Text</h4>
              <p className="text-sm text-gray-900 mt-1">OpenAI Vision API for room feature extraction</p>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <Search className="w-8 h-8 text-red-600 mx-auto mb-2" />
              <h4 className="font-medium text-gray-900">Keyword Search</h4>
              <p className="text-sm text-gray-900 mt-1">Traditional text-based filtering system</p>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <Filter className="w-8 h-8 text-red-600 mx-auto mb-2" />
              <h4 className="font-medium text-gray-900">Vector Search</h4>
              <p className="text-sm text-gray-900 mt-1">Semantic similarity matching</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

