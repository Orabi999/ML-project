import React, { useState } from 'react';
import locationsData from '../locations.json';
import { PredictionRequest, PredictionResponse } from '../types/prediction';

export const PredictionForm: React.FC = () => {
  const [formData, setFormData] = useState<PredictionRequest>({
    location: locationsData[0] || 'other',
    carpet_area_sqft: 1000,
    floor_num: 1,
    bathroom: 2,
    balcony: 1,
    furnishing: 'Semi-Furnished',
    transaction: 'Resale',
    ownership: 'Freehold',
    facing: 'North',
  });

  const [result, setResult] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    try {
      const baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
      const response = await fetch(`${baseUrl}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!response.ok) throw new Error('Failed to get prediction from server.');
      
      const data: PredictionResponse = await response.json();
      setResult(data.predicted_price);
    } catch (err: any) {
      setError(err.message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  const formatPrice = (price: number) => {
    if (price >= 1e7) return `₹ ${(price / 1e7).toFixed(2)} Cr`;
    if (price >= 1e5) return `₹ ${(price / 1e5).toFixed(2)} Lac`;
    return `₹ ${price.toLocaleString()}`;
  };

  return (
    <div style={{ maxWidth: '500px', margin: '2rem auto', fontFamily: 'sans-serif', padding: '1rem', border: '1px solid #ccc', borderRadius: '8px' }}>
      <h2 style={{ textAlign: 'center' }}>House Price Predictor</h2>
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
          Location:
          <select value={formData.location} onChange={(e) => setFormData({ ...formData, location: e.target.value })}>
            {locationsData.map((loc) => (
              <option key={loc} value={loc}>{loc}</option>
            ))}
          </select>
        </label>

        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
          Carpet Area (sqft):
          <input
            type="number"
            min="1"
            value={formData.carpet_area_sqft}
            onChange={(e) => setFormData({ ...formData, carpet_area_sqft: Number(e.target.value) })}
          />
        </label>

        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
          Floor Number:
          <input
            type="number"
            min="0"
            value={formData.floor_num}
            onChange={(e) => setFormData({ ...formData, floor_num: Number(e.target.value) })}
          />
        </label>

        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
          Bathrooms:
          <input
            type="number"
            min="1"
            value={formData.bathroom}
            onChange={(e) => setFormData({ ...formData, bathroom: Number(e.target.value) })}
          />
        </label>

        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
          Balconies:
          <input
            type="number"
            min="0"
            value={formData.balcony}
            onChange={(e) => setFormData({ ...formData, balcony: Number(e.target.value) })}
          />
        </label>

        <button type="submit" disabled={loading} style={{ padding: '0.6rem', cursor: 'pointer', background: '#007bff', color: '#fff', border: 'none', borderRadius: '4px' }}>
          {loading ? 'Predicting...' : 'Get Estimated Price'}
        </button>
      </form>

      {error && <p style={{ color: 'red', marginTop: '1rem' }}>{error}</p>}
      {result !== null && (
        <div style={{ marginTop: '1.5rem', padding: '1rem', background: '#e0f7fa', borderRadius: '8px', textAlign: 'center' }}>
          <h3>Estimated Price:</h3>
          <p style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#006064' }}>{formatPrice(result)}</p>
        </div>
      )}
    </div>
  );
};
