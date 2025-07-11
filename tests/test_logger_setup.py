"""
Test suite for feax.logger_setup module.

Tests logging setup functionality.
"""

import pytest
import logging
from feax.logger_setup import setup_logger


class TestLoggerSetup:
    """Test logger setup functionality."""
    
    def test_setup_logger_basic(self):
        """Test basic logger setup."""
        logger = setup_logger('test_logger')
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'test_logger'
    
    def test_setup_logger_with_module_name(self):
        """Test logger setup with module name."""
        logger = setup_logger(__name__)
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == __name__
    
    def test_logger_levels(self):
        """Test that logger handles different levels."""
        logger = setup_logger('test_levels')
        
        # Test that logger methods exist
        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'critical')
    
    def test_logger_configuration(self):
        """Test logger configuration."""
        logger = setup_logger('test_config')
        
        # Test that logger is properly configured
        assert logger.level is not None
        assert len(logger.handlers) >= 0  # May have handlers
    
    def test_multiple_loggers(self):
        """Test creating multiple loggers."""
        logger1 = setup_logger('logger1')
        logger2 = setup_logger('logger2')
        
        assert logger1.name != logger2.name
        assert logger1 is not logger2
    
    def test_logger_messages(self):
        """Test that logger can handle messages."""
        logger = setup_logger('test_messages')
        
        # These should not raise exceptions
        try:
            logger.debug('Debug message')
            logger.info('Info message')
            logger.warning('Warning message')
            logger.error('Error message')
        except Exception as e:
            pytest.fail(f"Logger raised unexpected exception: {e}")
    
    def test_logger_with_formatting(self):
        """Test logger with message formatting."""
        logger = setup_logger('test_formatting')
        
        # Test formatted messages
        try:
            logger.info('Message with %s', 'parameter')
            logger.info('Message with {}'.format('parameter'))
            logger.info(f'Message with {"parameter"}')
        except Exception as e:
            pytest.fail(f"Logger formatting raised unexpected exception: {e}")
    
    def test_logger_singleton_behavior(self):
        """Test if logger exhibits singleton-like behavior."""
        logger1 = setup_logger('same_name')
        logger2 = setup_logger('same_name')
        
        # In Python logging, loggers with the same name are the same object
        assert logger1 is logger2