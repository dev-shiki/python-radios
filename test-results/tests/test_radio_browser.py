import asyncio
import socket
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
    session.request = AsyncMock()
    return session


@pytest.fixture
def mock_response():
    """Return a mock response."""
    response = AsyncMock()
    response.status = 200
    response.headers = {"Content-Type": "application/json"}
    response.text = AsyncMock(return_value='{"key": "value"}')
    return response


@pytest.fixture
def mock_dns_resolver():
    """Return a mock DNSResolver."""
    resolver = AsyncMock()
    result = [MagicMock()]
    result[0].host = "api.radio-browser.info"
    resolver.query.return_value = result
    return resolver


@pytest.fixture
def stats_data():
    """Return stats data."""
    return {
        "supported_version": 1,
        "software_version": "1.0.0",
        "status": "OK",
        "stations": 1000,
        "stations_broken": 10,
        "tags": 500,
        "clicks_last_hour": 100,
        "clicks_last_day": 1000,
        "languages": 50,
        "countries": 100,
    }


@pytest.fixture
def country_data():
    """Return country data."""
    return {
        "name": "US",
        "stationcount": "1000",
    }


@pytest.fixture
def language_data():
    """Return language data."""
    return {
        "name": "english",
        "stationcount": "1000",
        "iso_639": "en",
    }


@pytest.fixture
def station_data():
    """Return station data."""
    return {
        "changeuuid": "12345",
        "stationuuid": "abcde",
        "name": "Test Station",
        "url": "https://example.com/stream",
        "url_resolved": "https://example.com/stream",
        "homepage": "https://example.com",
        "favicon": "https://example.com/favicon.ico",
        "tags": "test,radio",
        "country": "United States",
        "countrycode": "US",
        "state": "California",
        "language": "English",
        "languagecodes": "en",
        "votes": 10,
        "lastchangetime_iso8601": "2023-01-01T00:00:00Z",
        "lastchecktime_iso8601": "2023-01-01T00:00:00Z",
        "lastcheckoktime_iso8601": "2023-01-01T00:00:00Z",
        "lastlocalchecktime_iso8601": "2023-01-01T00:00:00Z",
        "clicktimestamp_iso8601": "2023-01-01T00:00:00Z",
        "clickcount": 100,
        "clicktrend": 5,
        "ssl_error": 0,
        "geo_lat": 37.7749,
        "geo_long": -122.4194,
        "has_extended_info": True,
        "codec": "MP3",
        "bitrate": 128,
        "hls": False,
        "lastcheckok": True,
        "iso_3166_2": "US-CA",
    }


@pytest.fixture
def tag_data():
    """Return tag data."""
    return {
        "name": "rock",
        "stationcount": "500",
    }


@pytest.mark.asyncio
async def test_request_success(radio_browser, mock_session, mock_response, mock_dns_resolver):
    """Test successful request."""
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        radio_browser.session = mock_session
        mock_session.request.return_value = mock_response

        result = await radio_browser._request("test")

        assert result == '{"key": "value"}'
        mock_dns_resolver.query.assert_called_once_with("_api._tcp.radio-browser.info", "SRV")
        mock_session.request.assert_called_once()


@pytest.mark.asyncio
async def test_request_creates_session_if_none(radio_browser, mock_response, mock_dns_resolver):
    """Test request creates session if none exists."""
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver), \
         patch("radios.radio_browser.aiohttp.ClientSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session.request.return_value = mock_response
        mock_session_class.return_value = mock_session

        radio_browser.session = None
        await radio_browser._request("test")

        mock_session_class.assert_called_once()
        assert radio_browser._close_session is True


@pytest.mark.asyncio
async def test_request_timeout(radio_browser, mock_session, mock_dns_resolver):
    """Test request timeout."""
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        radio_browser.session = mock_session
        mock_session.request.side_effect = asyncio.TimeoutError()

        with pytest.raises(RadioBrowserConnectionTimeoutError):
            await radio_browser._request("test")

        assert radio_browser._host is None


@pytest.mark.asyncio
async def test_request_connection_error(radio_browser, mock_session, mock_dns_resolver):
    """Test request connection error."""
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        radio_browser.session = mock_session
        mock_session.request.side_effect = aiohttp.ClientError()

        with pytest.raises(RadioBrowserConnectionError):
            await radio_browser._request("test")

        assert radio_browser._host is None


@pytest.mark.asyncio
async def test_request_socket_error(radio_browser, mock_session, mock_dns_resolver):
    """Test request socket error."""
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        radio_browser.session = mock_session
        mock_session.request.side_effect = socket.gaierror()

        with pytest.raises(RadioBrowserConnectionError):
            await radio_browser._request("test")

        assert radio_browser._host is None


@pytest.mark.asyncio
async def test_request_dns_error(radio_browser, mock_session):
    """Test request DNS error."""
    with patch("radios.radio_browser.DNSResolver") as mock_dns_resolver_class:
        mock_dns_resolver = AsyncMock()
        mock_dns_resolver.query.side_effect = DNSError()
        mock_dns_resolver_class.return_value = mock_dns_resolver

        radio_browser.session = mock_session

        with pytest.raises(Exception):
            await radio_browser._request("test")


@pytest.mark.asyncio
async def test_request_invalid_content_type(radio_browser, mock_session, mock_dns_resolver):
    """Test request with invalid content type."""
    with patch("radios.radio_browser.DNSResolver", return_value=mock_dns_resolver):
        radio_browser.session = mock_session
        
        response = AsyncMock()
        response.status = 200
        response.headers = {"Content-Type": "text/html"}
        response.text = AsyncMock(return_value="<html>Not JSON</html>")
        mock_session.request.return_value = response

        with pytest.raises(RadioBrowserError):
            await radio_browser._request("test")


@pytest.mark.asyncio
async def test_stats(radio_browser, stats_data):
    """Test getting stats."""
    with patch.object(radio_browser, "_request") as mock_request:
        mock_request.return_value = orjson.dumps(stats_data)
        
        result = await radio_browser.stats()
        
        mock_request.assert_called_once_with("stats")
        assert isinstance(result, Stats)
        assert result.supported_version == stats_data["supported_version"]
        assert result.software_version == AwesomeVersion(stats_data["software_version"])
        assert result.status == stats_data["status"]
        assert result.stations == stats_data["stations"]
        assert result.stations_broken == stats_data["stations_broken"]
        assert result.tags == stats_data["tags"]
        assert result.clicks_last_hour == stats_data["clicks_last_hour"]
        assert result.clicks_last_day == stats_data["clicks_last_day"]
        assert result.languages == stats_data["languages"]
        assert result.countries == stats_data["countries"]


@pytest.mark.asyncio
async def test_station_click(radio_browser):
    """Test station click."""
    with patch.object(radio_browser, "_request") as mock_request:
        mock_request.return_value = "{}"
        
        await radio_browser.station_click(uuid="test-uuid")
        
        mock_request.assert_called_once_with("url/test-uuid")


@pytest.mark.asyncio
async def test_countries(radio_browser, country_data):
    """Test getting countries."""
    with patch.object(radio_browser, "_request") as mock_request:
        # Add code field to match what the API returns
        country_data_with_code = country_data.copy()
        country_data_with_code["code"] = country_data["name"]
        mock_request.return_value = orjson.dumps([country_data_with_code])
        
        result = await radio_browser.countries()
        
        mock_request.assert_called_once_with(
            "countrycodes",
            params={
                "hidebroken": False,
                "limit": 100000,
                "offset": 0,
                "order": "name",
                "reverse": False,
            },
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Country)
        assert result[0].code == country_data["name"]
        assert result[0].station_count == country_data["stationcount"]


@pytest.mark.asyncio
async def test_countries_with_kosovo(radio_browser):
    """Test getting countries with Kosovo special case."""
    with patch.object(radio_browser, "_request") as mock_request:
        kosovo_data = {"name": "XK", "stationcount": "100"}
        mock_request.return_value = orjson.dumps([kosovo_data])
        
        result = await radio_browser.countries()
        
        assert len(result) == 1
        assert result[0].name == "Kosovo"
        assert result[0].code == "XK"


@pytest.mark.asyncio
async def test_countries_with_parameters(radio_browser):
    """Test getting countries with custom parameters."""
    with patch.object(radio_browser, "_request") as mock_request:
        mock_request.return_value = orjson.dumps([])
        
        await radio_browser.countries(
            hide_broken=True,
            limit=10,
            offset=5,
            order=Order.STATION_COUNT,
            reverse=True,
        )
        
        mock_request.assert_called_once_with(
            "countrycodes",
            params={
                "hidebroken": True,
                "limit": 10,
                "offset": 5,
                "order": "stationcount",
                "reverse": True,
            },
        )


@pytest.mark.asyncio
async def test_languages(radio_browser, language_data):
    """Test getting languages."""
    with patch.object(radio_browser, "_request") as mock_request:
        mock_request.return_value = orjson.dumps([language_data])
        
        result = await radio_browser.languages()
        
        mock_request.assert_called_once_with(
            "languages",
            params={
                "hidebroken": False,
                "limit": 100000,
                "offset": 0,
                "order": "name",
                "reverse": False,
            },
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Language)
        assert result[0].name == language_data["name"].title()
        assert result[0].code == language_data["iso_639"]
        assert result[0].station_count == language_data["stationcount"]


@pytest.mark.asyncio
async def test_languages_with_parameters(radio_browser):
    """Test getting languages with custom parameters."""
    with patch.object(radio_browser, "_request") as mock_request:
        mock_request.return_value = orjson.dumps([])
        
        await radio_browser.languages(
            hide_broken=True,
            limit=10,
            offset=5,
            order=Order.STATION_COUNT,
            reverse=True,
        )
        
        mock_request.assert_called_once_with(
            "languages",
            params={
                "hidebroken": True,
                "limit": 10,
                "offset": 5,
                "order": "stationcount",
                "reverse": True,
            },
        )


@pytest.mark.asyncio
async def test_search(radio_browser, station_data):
    """Test search for stations."""
    with patch.object(radio_browser, "_request") as mock_request:
        mock_request.return_value = orjson.dumps([station_data])
        
        result = await radio_browser.search(name="test")
        
        mock_request.assert_called_once_with(
            "stations/search",
            params={
                "hidebroken": False,
                "limit": 100000,
                "offset": 0,
                "order": "name",
                "reverse": False,
                "name": "test",
                "name_exact": False,
                "country": "",
                "country_exact": False,
                "state_exact": False,
                "language_exact": False,
                "tag_exact": False,
                "bitrate_min": 0,
                "bitrate_max": 1000000,
            },
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Station)
        assert result[0].name == station_data["name"]
        assert result[0].uuid == station_data["stationuuid"]


@pytest.mark.asyncio
async def test_search_with_filter(radio_browser, station_data):
    """Test search with filter."""
    with patch.object(radio_browser, "_request") as mock_request:
        mock_request.return_value = orjson.dumps([station_data])
        
        result = await radio_browser.search(
            filter_by=FilterBy.COUNTRY,
            filter_term="US",
        )
        
        mock_request.assert_called_once_with(
            "stations/search/bycountry/US",
            params={
                "hidebroken": False,
                "limit": 100000,
                "offset": 0,
                "order": "name",
                "reverse": False,
                "name": None,
                "name_exact": False,
                "country": "",
                "country_exact": False,
                "state_exact": False,
                "language_exact": False,
                "tag_exact": False,
                "bitrate_min": 0,
                "bitrate_max": 1000000,
            },
        )
        assert len(result) == 1


@pytest.mark.asyncio
async def test_search_with_all_parameters(radio_browser):
    """Test search with all parameters."""
    with patch.object(radio_browser, "_request") as mock_request:
        mock_request.return_value = orjson.dumps([])
        
        await radio_browser.search(
            filter_by=FilterBy.TAG,
            filter_term="rock",
            hide_broken=True,
            limit=10,
            offset=5,
            order=Order.BITRATE,
            reverse=True,
            name="test",
            name_exact=True,
            country="US",
            country_exact=True,
            state_exact=True,
            language_exact=True,
            tag_exact=True,
            bitrate_min=128,
            bitrate_max=320,
        )
        
        mock_request.assert_called_once_with(
            "stations/search/bytag/rock",
            params={
                "hidebroken": True,
                "limit": 10,
                "offset": 5,
                "order": "bitrate",
                "reverse": True,
                "name": "test",
                "name_exact": True,
                "country": "US",
                "country_exact": True,
                "state_exact": True,
                "language_exact": True,
                "tag_exact": True,
                "bitrate_min": 128,
                "bitrate_max": 320,
            },
        )


@pytest.mark.asyncio
async def test_station_found(radio_browser, station_data):
    """Test getting a station by UUID when found."""
    with patch.object(radio_browser, "stations") as mock_stations:
        station = Station.from_dict(station_data)
        mock_stations.return_value = [station]
        
        result = await radio_browser.station(uuid="test-uuid")
        
        mock_stations.assert_called_once_with(
            filter_by=FilterBy.UUID,
            filter_term="test-uuid",
            limit=1,
        )
        assert result == station


@pytest.mark.asyncio
async def test_station_not_found(radio_browser):
    """Test getting a station by UUID when not found."""
    with patch.object(radio_browser, "stations") as mock_stations:
        mock_stations.return_value = []
        
        result = await radio_browser.station(uuid="test-uuid")
        
        mock_stations.assert_called_once_with(
            filter_by=FilterBy.UUID,
            filter_term="test-uuid",
            limit=1,
        )
        assert result is None


@pytest.mark.asyncio
async def test_stations(radio_browser, station_data):
    """Test getting stations."""
    with patch.object(radio_browser, "_request") as mock_request:
        mock_request.return_value = orjson.dumps([station_data])
        
        result = await radio_browser.stations()
        
        mock_request.assert_called_once_with(
            "stations",
            params={
                "hidebroken": False,
                "limit": 100000,
                "offset": 0,
                "order": "name",
                "reverse": False,
            },
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Station)
        assert result[0].name == station_data["name"]
        assert result[0].uuid == station_data["stationuuid"]


@pytest.mark.asyncio
async def test_stations_with_filter(radio_browser, station_data):
    """Test getting stations with filter."""
    with patch.object(radio_browser, "_request") as mock_request:
        mock_request.return_value = orjson.dumps([station_data])
        
        result = await radio_browser.stations(
            filter_by=FilterBy.COUNTRY,
            filter_term="US",
        )
        
        mock_request.assert_called_once_with(
            "stations/bycountry/US",
            params={
                "hidebroken": False,
                "limit": 100000,
                "offset": 0,
                "order": "name",
                "reverse": False,
            },
        )
        assert len(result) == 1


@pytest.mark.asyncio
async def test_stations_with_parameters(radio_browser):
    """Test getting stations with custom parameters."""
    with patch.object(radio_browser, "_request") as mock_request:
        mock_request.return_value = orjson.dumps([])
        
        await radio_browser.stations(
            hide_broken=True,
            limit=10,
            offset=5,
            order=Order.BITRATE,
            reverse=True,
        )
        
        mock_request.assert_called_once_with(
            "stations",
            params={
                "hidebroken": True,
                "limit": 10,
                "offset": 5,
                "order": "bitrate",
                "reverse": True,
            },
        )


@pytest.mark.asyncio
async def test_tags(radio_browser, tag_data):
    """Test getting tags."""
    with patch.object(radio_browser, "_request") as mock_request:
        mock_request.return_value = orjson.dumps([tag_data])
        
        result = await radio_browser.tags()
        
        mock_request.assert_called_once_with(
            "tags",
            params={
                "hidebroken": False,
                "limit": 100000,
                "offset": 0,
                "order": "name",
                "reverse": False,
            },
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Tag)
        assert result[0].name == tag_data["name"]
        assert result[0].station_count == tag_data["stationcount"]


@pytest.mark.asyncio
async def test_tags_with_parameters(radio_browser):
    """Test getting tags with custom parameters."""
    with patch.object(radio_browser, "_request") as mock_request:
        mock_request.return_value = orjson.dumps([])
        
        await radio_browser.tags(
            hide_broken=True,
            limit=10,
            offset=5,
            order=Order.STATION_COUNT,
            reverse=True,
        )
        
        mock_request.assert_called_once_with(
            "tags",
            params={
                "hidebroken": True,
                "limit": 10,
                "offset": 5,
                "order": "stationcount",
                "reverse": True,
            },
        )


@pytest.mark.asyncio
async def test_close(radio_browser):
    """Test closing the session."""
    radio_browser.session = AsyncMock()
    radio_browser._close_session = True
    
    await radio_browser.close()
    
    radio_browser.session.close.assert_called_once()


@pytest.mark.asyncio
async def test_close_no_session(radio_browser):
    """Test closing when no session exists."""
    radio_browser.session = None
    
    await radio_browser.close()  # Should not raise


@pytest.mark.asyncio
async def test_close_not_owned(radio_browser):
    """Test not closing session when not owned."""
    radio_browser.session = AsyncMock()
    radio_browser._close_session = False
    
    await radio_browser.close()
    
    radio_browser.session.close.assert_not_called()


@pytest.mark.asyncio
async def test_aenter(radio_browser):
    """Test async enter."""
    result = await radio_browser.__aenter__()
    assert result is radio_browser


@pytest.mark.asyncio
async def test_aexit(radio_browser):
    """Test async exit."""
    with patch.object(radio_browser, "close") as mock_close:
        await radio_browser.__aexit__(None, None, None)
        mock_close.assert_called_once()


@pytest.mark.asyncio
async def test_context_manager():
    """Test using RadioBrowser as a context manager."""
    with patch("radios.radio_browser.RadioBrowser.close") as mock_close:
        async with RadioBrowser(user_agent="Test/1.0") as rb:
            assert isinstance(rb, RadioBrowser)
        mock_close.assert_called_once()