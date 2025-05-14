import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import orjson
import pytest
from aiodns.error import DNSError
from awesomeversion import AwesomeVersion

from radios.const import FilterBy, Order
from radios.exceptions import (
    RadioBrowserConnectionError,
    RadioBrowserConnectionTimeoutError,
    RadioBrowserError,
)
from radios.models import Country, Language, Station, Stats, Tag
from radios.radio_browser import RadioBrowser


@pytest.fixture
def radio_browser():
    """Return a RadioBrowser instance."""
    return RadioBrowser(user_agent="Test/1.0")


@pytest.fixture
def mock_session():
    """Return a mock aiohttp ClientSession."""
    session = AsyncMock()
    response = AsyncMock()
    response.status = 200
    response.headers = {"Content-Type": "application/json"}
    response.text = AsyncMock(return_value='{"test": "data"}')
    session.request.return_value = response
    return session


@pytest.fixture
def mock_dns_resolver():
    """Return a mock DNSResolver."""
    resolver = AsyncMock()
    result = [MagicMock(host="api.radio-browser.info")]
    resolver.query.return_value = result
    return resolver


@pytest.mark.asyncio
async def test_request_dns_resolution(radio_browser, mock_dns_resolver):
    """Test DNS resolution in _request method."""
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        with patch.object(radio_browser, "session", AsyncMock()) as mock_session:
            response = AsyncMock()
            response.status = 200
            response.headers = {"Content-Type": "application/json"}
            response.text = AsyncMock(return_value='{"test": "data"}')
            mock_session.request.return_value = response
            
            await radio_browser._request("test")
            
            mock_dns_resolver.query.assert_called_once_with(
                "_api._tcp.radio-browser.info", "SRV"
            )
            assert radio_browser._host == "api.radio-browser.info"


@pytest.mark.asyncio
async def test_request_creates_session_if_none(radio_browser):
    """Test that _request creates a session if none exists."""
    with patch("radios.radio_browser.aiohttp.ClientSession") as mock_session_class:
        mock_session = AsyncMock()
        response = AsyncMock()
        response.status = 200
        response.headers = {"Content-Type": "application/json"}
        response.text = AsyncMock(return_value='{"test": "data"}')
        mock_session.request.return_value = response
        mock_session_class.return_value = mock_session
        
        with patch("radios.radio_browser.DNSResolver") as mock_resolver_class:
            mock_resolver = AsyncMock()
            mock_result = [MagicMock(host="api.radio-browser.info")]
            mock_resolver.query.return_value = mock_result
            mock_resolver_class.return_value = mock_resolver
            
            await radio_browser._request("test")
            
            mock_session_class.assert_called_once()
            assert radio_browser.session is not None
            assert radio_browser._close_session is True


@pytest.mark.asyncio
async def test_request_timeout(radio_browser, mock_dns_resolver):
    """Test timeout handling in _request method."""
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        with patch.object(radio_browser, "session", AsyncMock()) as mock_session:
            mock_session.request.side_effect = asyncio.TimeoutError()
            
            with pytest.raises(RadioBrowserConnectionTimeoutError):
                await radio_browser._request("test")
            
            assert radio_browser._host is None


@pytest.mark.asyncio
async def test_request_connection_error(radio_browser, mock_dns_resolver):
    """Test connection error handling in _request method."""
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        with patch.object(radio_browser, "session", AsyncMock()) as mock_session:
            mock_session.request.side_effect = aiohttp.ClientError()
            
            with pytest.raises(RadioBrowserConnectionError):
                await radio_browser._request("test")
            
            assert radio_browser._host is None


@pytest.mark.asyncio
async def test_request_socket_error(radio_browser, mock_dns_resolver):
    """Test socket error handling in _request method."""
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        with patch.object(radio_browser, "session", AsyncMock()) as mock_session:
            mock_session.request.side_effect = socket.gaierror()
            
            with pytest.raises(RadioBrowserConnectionError):
                await radio_browser._request("test")
            
            assert radio_browser._host is None


@pytest.mark.asyncio
async def test_request_invalid_content_type(radio_browser, mock_dns_resolver):
    """Test invalid content type handling in _request method."""
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        with patch.object(radio_browser, "session", AsyncMock()) as mock_session:
            response = AsyncMock()
            response.status = 200
            response.headers = {"Content-Type": "text/html"}
            response.text = AsyncMock(return_value="<html>Not JSON</html>")
            mock_session.request.return_value = response
            
            with pytest.raises(RadioBrowserError):
                await radio_browser._request("test")


@pytest.mark.asyncio
async def test_request_boolean_params(radio_browser, mock_dns_resolver):
    """Test boolean parameter handling in _request method."""
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        with patch.object(radio_browser, "session", AsyncMock()) as mock_session:
            response = AsyncMock()
            response.status = 200
            response.headers = {"Content-Type": "application/json"}
            response.text = AsyncMock(return_value='{"test": "data"}')
            mock_session.request.return_value = response
            
            await radio_browser._request("test", params={"bool_param": True})
            
            mock_session.request.assert_called_once()
            call_args = mock_session.request.call_args[1]
            assert call_args["params"]["bool_param"] == "true"


@pytest.mark.asyncio
async def test_stats(radio_browser):
    """Test stats method."""
    stats_json = """
    {
        "supported_version": 1,
        "software_version": "1.0.0",
        "status": "OK",
        "stations": 1000,
        "stations_broken": 10,
        "tags": 500,
        "clicks_last_hour": 100,
        "clicks_last_day": 2400,
        "languages": 50,
        "countries": 100
    }
    """
    
    with patch.object(radio_browser, "_request", AsyncMock(return_value=stats_json)):
        result = await radio_browser.stats()
        
        assert isinstance(result, Stats)
        assert result.supported_version == 1
        assert result.software_version == AwesomeVersion("1.0.0")
        assert result.status == "OK"
        assert result.stations == 1000
        assert result.stations_broken == 10
        assert result.tags == 500
        assert result.clicks_last_hour == 100
        assert result.clicks_last_day == 2400
        assert result.languages == 50
        assert result.countries == 100


@pytest.mark.asyncio
async def test_station_click(radio_browser):
    """Test station_click method."""
    with patch.object(radio_browser, "_request", AsyncMock()) as mock_request:
        await radio_browser.station_click(uuid="test-uuid")
        
        mock_request.assert_called_once_with("url/test-uuid")


@pytest.mark.asyncio
async def test_countries(radio_browser):
    """Test countries method."""
    countries_json = """
    [
        {"name": "US", "stationcount": "500"},
        {"name": "GB", "stationcount": "300"},
        {"name": "XK", "stationcount": "50"}
    ]
    """
    
    with patch.object(radio_browser, "_request", AsyncMock(return_value=countries_json)):
        result = await radio_browser.countries()
        
        assert len(result) == 3
        assert isinstance(result[0], Country)
        
        # Find countries by code
        us = next(c for c in result if c.code == "US")
        gb = next(c for c in result if c.code == "GB")
        xk = next(c for c in result if c.code == "XK")
        
        assert us.name == "United States"
        assert us.station_count == "500"
        assert us.favicon == "https://flagcdn.com/256x192/us.png"
        
        assert gb.name == "United Kingdom"
        assert gb.station_count == "300"
        assert gb.favicon == "https://flagcdn.com/256x192/gb.png"
        
        # Test Kosovo special case
        assert xk.name == "Kosovo"
        assert xk.station_count == "50"
        assert xk.favicon == "https://flagcdn.com/256x192/xk.png"


@pytest.mark.asyncio
async def test_countries_with_parameters(radio_browser):
    """Test countries method with parameters."""
    with patch.object(radio_browser, "_request", AsyncMock(return_value="[]")):
        await radio_browser.countries(
            hide_broken=True,
            limit=10,
            offset=5,
            order=Order.STATION_COUNT,
            reverse=True
        )
        
        radio_browser._request.assert_called_once()
        call_args = radio_browser._request.call_args[1]
        assert call_args["params"]["hidebroken"] is True
        assert call_args["params"]["limit"] == 10
        assert call_args["params"]["offset"] == 5
        assert call_args["params"]["order"] == "stationcount"
        assert call_args["params"]["reverse"] is True


@pytest.mark.asyncio
async def test_languages(radio_browser):
    """Test languages method."""
    languages_json = """
    [
        {"name": "english", "stationcount": "500", "iso_639": "en"},
        {"name": "spanish", "stationcount": "300", "iso_639": "es"},
        {"name": "french", "stationcount": "200", "iso_639": "fr"}
    ]
    """
    
    with patch.object(radio_browser, "_request", AsyncMock(return_value=languages_json)):
        result = await radio_browser.languages()
        
        assert len(result) == 3
        assert isinstance(result[0], Language)
        
        # Find languages by name
        english = next(l for l in result if l.name == "English")
        spanish = next(l for l in result if l.name == "Spanish")
        french = next(l for l in result if l.name == "French")
        
        assert english.code == "en"
        assert english.station_count == "500"
        assert english.favicon == "https://flagcdn.com/256x192/en.png"
        
        assert spanish.code == "es"
        assert spanish.station_count == "300"
        assert spanish.favicon == "https://flagcdn.com/256x192/es.png"
        
        assert french.code == "fr"
        assert french.station_count == "200"
        assert french.favicon == "https://flagcdn.com/256x192/fr.png"


@pytest.mark.asyncio
async def test_languages_with_parameters(radio_browser):
    """Test languages method with parameters."""
    with patch.object(radio_browser, "_request", AsyncMock(return_value="[]")):
        await radio_browser.languages(
            hide_broken=True,
            limit=10,
            offset=5,
            order=Order.STATION_COUNT,
            reverse=True
        )
        
        radio_browser._request.assert_called_once()
        call_args = radio_browser._request.call_args[1]
        assert call_args["params"]["hidebroken"] is True
        assert call_args["params"]["limit"] == 10
        assert call_args["params"]["offset"] == 5
        assert call_args["params"]["order"] == "stationcount"
        assert call_args["params"]["reverse"] is True


@pytest.mark.asyncio
async def test_search(radio_browser):
    """Test search method."""
    station_json = """
    [
        {
            "changeuuid": "change-uuid-1",
            "stationuuid": "station-uuid-1",
            "name": "Test Station 1",
            "url": "http://example.com/stream1",
            "url_resolved": "http://example.com/stream1",
            "homepage": "http://example.com",
            "favicon": "http://example.com/favicon.ico",
            "tags": "tag1,tag2",
            "country": "United States",
            "countrycode": "US",
            "state": "California",
            "language": "English",
            "languagecodes": "en",
            "votes": 10,
            "clickcount": 100,
            "clicktrend": 5,
            "lastchangetime_iso8601": "2023-01-01T12:00:00Z",
            "codec": "MP3",
            "bitrate": 128,
            "hls": false,
            "lastcheckok": true,
            "lastchecktime_iso8601": "2023-01-01T12:00:00Z",
            "lastcheckoktime_iso8601": "2023-01-01T12:00:00Z",
            "lastlocalchecktime_iso8601": "2023-01-01T12:00:00Z",
            "clicktimestamp_iso8601": "2023-01-01T12:00:00Z",
            "geo_lat": 37.7749,
            "geo_long": -122.4194,
            "has_extended_info": true,
            "iso_3166_2": "US-CA",
            "ssl_error": 0
        }
    ]
    """
    
    with patch.object(radio_browser, "_request", AsyncMock(return_value=station_json)):
        result = await radio_browser.search(
            filter_by=FilterBy.COUNTRY,
            filter_term="US",
            hide_broken=True,
            limit=10,
            offset=5,
            order=Order.NAME,
            reverse=False,
            name="Test",
            name_exact=False,
            country="United States",
            country_exact=True,
            state_exact=False,
            language_exact=False,
            tag_exact=False,
            bitrate_min=64,
            bitrate_max=320
        )
        
        assert len(result) == 1
        assert isinstance(result[0], Station)
        assert result[0].name == "Test Station 1"
        assert result[0].uuid == "station-uuid-1"
        assert result[0].url == "http://example.com/stream1"
        assert result[0].bitrate == 128
        assert result[0].country == "United States"
        
        radio_browser._request.assert_called_once()
        call_args = radio_browser._request.call_args
        assert call_args[0][0] == "stations/search/bycountry/US"
        params = call_args[1]["params"]
        assert params["hidebroken"] is True
        assert params["limit"] == 10
        assert params["offset"] == 5
        assert params["order"] == "name"
        assert params["reverse"] is False
        assert params["name"] == "Test"
        assert params["name_exact"] is False
        assert params["country"] == "United States"
        assert params["country_exact"] is True
        assert params["state_exact"] is False
        assert params["language_exact"] is False
        assert params["tag_exact"] is False
        assert params["bitrate_min"] == 64
        assert params["bitrate_max"] == 320


@pytest.mark.asyncio
async def test_search_without_filter(radio_browser):
    """Test search method without filter parameters."""
    with patch.object(radio_browser, "_request", AsyncMock(return_value="[]")):
        await radio_browser.search()
        
        radio_browser._request.assert_called_once()
        assert radio_browser._request.call_args[0][0] == "stations/search"


@pytest.mark.asyncio
async def test_station_found(radio_browser):
    """Test station method when station is found."""
    station_json = """
    [
        {
            "changeuuid": "change-uuid-1",
            "stationuuid": "test-uuid",
            "name": "Test Station",
            "url": "http://example.com/stream",
            "url_resolved": "http://example.com/stream",
            "homepage": "http://example.com",
            "favicon": "http://example.com/favicon.ico",
            "tags": "tag1,tag2",
            "country": "United States",
            "countrycode": "US",
            "state": "California",
            "language": "English",
            "languagecodes": "en",
            "votes": 10,
            "clickcount": 100,
            "clicktrend": 5,
            "lastchangetime_iso8601": "2023-01-01T12:00:00Z",
            "codec": "MP3",
            "bitrate": 128,
            "hls": false,
            "lastcheckok": true,
            "lastchecktime_iso8601": "2023-01-01T12:00:00Z",
            "lastcheckoktime_iso8601": "2023-01-01T12:00:00Z",
            "lastlocalchecktime_iso8601": "2023-01-01T12:00:00Z",
            "clicktimestamp_iso8601": "2023-01-01T12:00:00Z",
            "geo_lat": 37.7749,
            "geo_long": -122.4194,
            "has_extended_info": true,
            "iso_3166_2": "US-CA",
            "ssl_error": 0
        }
    ]
    """
    
    with patch.object(radio_browser, "stations", AsyncMock(return_value=[
        Station(
            change_uuid="change-uuid-1",
            uuid="test-uuid",
            name="Test Station",
            url="http://example.com/stream",
            url_resolved="http://example.com/stream",
            homepage="http://example.com",
            favicon="http://example.com/favicon.ico",
            tags=["tag1", "tag2"],
            country_code="US",
            state="California",
            language=["English"],
            language_codes=["en"],
            votes=10,
            click_count=100,
            click_trend=5,
            lastchange_time=datetime.fromisoformat("2023-01-01T12:00:00Z"),
            codec="MP3",
            bitrate=128,
            hls=False,
            lastcheckok=True,
            last_check_time=datetime.fromisoformat("2023-01-01T12:00:00Z"),
            last_check_ok_time=datetime.fromisoformat("2023-01-01T12:00:00Z"),
            last_local_check_time=datetime.fromisoformat("2023-01-01T12:00:00Z"),
            click_timestamp=datetime.fromisoformat("2023-01-01T12:00:00Z"),
            latitude=37.7749,
            longitude=-122.4194,
            has_extended_info=True,
            iso_3166_2="US-CA",
            ssl_error=0
        )
    ])):
        result = await radio_browser.station(uuid="test-uuid")
        
        assert result is not None
        assert result.uuid == "test-uuid"
        assert result.name == "Test Station"
        
        radio_browser.stations.assert_called_once_with(
            filter_by=FilterBy.UUID,
            filter_term="test-uuid",
            limit=1
        )


@pytest.mark.asyncio
async def test_station_not_found(radio_browser):
    """Test station method when station is not found."""
    with patch.object(radio_browser, "stations", AsyncMock(return_value=[])):
        result = await radio_browser.station(uuid="nonexistent-uuid")
        
        assert result is None
        
        radio_browser.stations.assert_called_once_with(
            filter_by=FilterBy.UUID,
            filter_term="nonexistent-uuid",
            limit=1
        )


@pytest.mark.asyncio
async def test_stations(radio_browser):
    """Test stations method."""
    station_json = """
    [
        {
            "changeuuid": "change-uuid-1",
            "stationuuid": "station-uuid-1",
            "name": "Test Station 1",
            "url": "http://example.com/stream1",
            "url_resolved": "http://example.com/stream1",
            "homepage": "http://example.com",
            "favicon": "http://example.com/favicon.ico",
            "tags": "tag1,tag2",
            "country": "United States",
            "countrycode": "US",
            "state": "California",
            "language": "English",
            "languagecodes": "en",
            "votes": 10,
            "clickcount": 100,
            "clicktrend": 5,
            "lastchangetime_iso8601": "2023-01-01T12:00:00Z",
            "codec": "MP3",
            "bitrate": 128,
            "hls": false,
            "lastcheckok": true,
            "lastchecktime_iso8601": "2023-01-01T12:00:00Z",
            "lastcheckoktime_iso8601": "2023-01-01T12:00:00Z",
            "lastlocalchecktime_iso8601": "2023-01-01T12:00:00Z",
            "clicktimestamp_iso8601": "2023-01-01T12:00:00Z",
            "geo_lat": 37.7749,
            "geo_long": -122.4194,
            "has_extended_info": true,
            "iso_3166_2": "US-CA",
            "ssl_error": 0
        }
    ]
    """
    
    with patch.object(radio_browser, "_request", AsyncMock(return_value=station_json)):
        result = await radio_browser.stations(
            filter_by=FilterBy.COUNTRY,
            filter_term="US",
            hide_broken=True,
            limit=10,
            offset=5,
            order=Order.NAME,
            reverse=True
        )
        
        assert len(result) == 1
        assert isinstance(result[0], Station)
        assert result[0].name == "Test Station 1"
        assert result[0].uuid == "station-uuid-1"
        
        radio_browser._request.assert_called_once()
        call_args = radio_browser._request.call_args
        assert call_args[0][0] == "stations/bycountry/US"
        params = call_args[1]["params"]
        assert params["hidebroken"] is True
        assert params["limit"] == 10
        assert params["offset"] == 5
        assert params["order"] == "name"
        assert params["reverse"] is True


@pytest.mark.asyncio
async def test_stations_without_filter(radio_browser):
    """Test stations method without filter parameters."""
    with patch.object(radio_browser, "_request", AsyncMock(return_value="[]")):
        await radio_browser.stations()
        
        radio_browser._request.assert_called_once()
        assert radio_browser._request.call_args[0][0] == "stations"


@pytest.mark.asyncio
async def test_tags(radio_browser):
    """Test tags method."""
    tags_json = """
    [
        {"name": "rock", "stationcount": "500"},
        {"name": "pop", "stationcount": "300"},
        {"name": "jazz", "stationcount": "200"}
    ]
    """
    
    with patch.object(radio_browser, "_request", AsyncMock(return_value=tags_json)):
        result = await radio_browser.tags()
        
        assert len(result) == 3
        assert isinstance(result[0], Tag)
        
        # Find tags by name
        rock = next(t for t in result if t.name == "rock")
        pop = next(t for t in result if t.name == "pop")
        jazz = next(t for t in result if t.name == "jazz")
        
        assert rock.station_count == "500"
        assert pop.station_count == "300"
        assert jazz.station_count == "200"


@pytest.mark.asyncio
async def test_tags_with_parameters(radio_browser):
    """Test tags method with parameters."""
    with patch.object(radio_browser, "_request", AsyncMock(return_value="[]")):
        await radio_browser.tags(
            hide_broken=True,
            limit=10,
            offset=5,
            order=Order.STATION_COUNT,
            reverse=True
        )
        
        radio_browser._request.assert_called_once()
        call_args = radio_browser._request.call_args[1]
        assert call_args["params"]["hidebroken"] is True
        assert call_args["params"]["limit"] == 10
        assert call_args["params"]["offset"] == 5
        assert call_args["params"]["order"] == "stationcount"
        assert call_args["params"]["reverse"] is True


@pytest.mark.asyncio
async def test_close(radio_browser):
    """Test close method."""
    # Set up a session that should be closed
    radio_browser.session = AsyncMock()
    radio_browser._close_session = True
    
    await radio_browser.close()
    
    radio_browser.session.close.assert_called_once()


@pytest.mark.asyncio
async def test_close_no_session(radio_browser):
    """Test close method when no session exists."""
    radio_browser.session = None
    
    await radio_browser.close()  # Should not raise any exceptions


@pytest.mark.asyncio
async def test_close_external_session(radio_browser):
    """Test close method with external session."""
    # Set up a session that should not be closed
    radio_browser.session = AsyncMock()
    radio_browser._close_session = False
    
    await radio_browser.close()
    
    radio_browser.session.close.assert_not_called()


@pytest.mark.asyncio
async def test_aenter(radio_browser):
    """Test __aenter__ method."""
    result = await radio_browser.__aenter__()
    
    assert result is radio_browser


@pytest.mark.asyncio
async def test_aexit(radio_browser):
    """Test __aexit__ method."""
    with patch.object(radio_browser, "close", AsyncMock()) as mock_close:
        await radio_browser.__aexit__(None, None, None)
        
        mock_close.assert_called_once()


@pytest.mark.asyncio
async def test_async_context_manager(radio_browser):
    """Test using RadioBrowser as an async context manager."""
    with patch.object(radio_browser, "close", AsyncMock()) as mock_close:
        async with radio_browser as rb:
            assert rb is radio_browser
        
        mock_close.assert_called_once()


@pytest.mark.asyncio
async def test_dns_resolution_error(radio_browser):
    """Test handling of DNS resolution errors."""
    with patch("radios.radio_browser.DNSResolver") as mock_resolver_class:
        mock_resolver = AsyncMock()
        mock_resolver.query.side_effect = DNSError()
        mock_resolver_class.return_value = mock_resolver
        
        with pytest.raises(RadioBrowserConnectionError):
            await radio_browser._request("test")