use ash::{vk, LoadingError};
use png::DecodingError;
use std::io;
use std::{error::Error, fmt::Display};
use tobj::LoadError;
use winit::error::OsError;

// All errors in this application are one of these variants, or can be transfered to one of these.
#[derive(Debug)]
pub enum AppError {
    EngineError(String),
    VulkanError(vk::Result),
}

impl Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl Error for AppError {}

impl From<OsError> for AppError {
    fn from(value: winit::error::OsError) -> Self {
        Self::EngineError(value.to_string())
    }
}

impl From<vk::Result> for AppError {
    fn from(value: vk::Result) -> Self {
        Self::VulkanError(value)
    }
}

impl From<LoadingError> for AppError {
    fn from(value: ash::LoadingError) -> Self {
        Self::EngineError(value.to_string())
    }
}

impl From<SuitabilityError> for AppError {
    fn from(value: SuitabilityError) -> Self {
        Self::EngineError(format!("Suitability error: {}", value.0))
    }
}

impl From<(Vec<vk::Pipeline>, vk::Result)> for AppError {
    fn from(value: (Vec<vk::Pipeline>, vk::Result)) -> Self {
        Self::VulkanError(value.1)
    }
}

impl From<io::Error> for AppError {
    fn from(value: io::Error) -> Self {
        Self::EngineError(value.to_string())
    }
}

impl From<DecodingError> for AppError {
    fn from(value: DecodingError) -> Self {
        Self::EngineError(value.to_string())
    }
}

impl From<LoadError> for AppError {
    fn from(value: LoadError) -> Self {
        Self::EngineError(format!("{:?}", value))
    }
}

#[derive(Debug)]
pub struct SuitabilityError(pub String);

impl Display for SuitabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for SuitabilityError {}
