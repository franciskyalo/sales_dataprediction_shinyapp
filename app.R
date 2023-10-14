library(shiny)
library(shinythemes)
library(DT)
library(tidyverse)

# load the model
model <- read_rds("salesmodel.rds")

# Define UI for application that draws a histogram
ui <- fluidPage(
  
  # Application title
  titlePanel("Sales Predictions"),
  theme = shinytheme("sandstone"),
  # Sidebar layout
  sidebarLayout(
    sidebarPanel(
      p("Dazzle your data❗️ Pop it in and get your predictions complete with a downloadable csv ⚡️"),
      br(),
      br(),
      br(),
      br(),
      # Input: Select a file ----
      fileInput("file1", "upload csv file here",
                multiple = FALSE,
                accept = c("text/csv",
                           "text/comma-separated-values,text/plain",
                           ".csv")),
      
      
      # Button
      downloadButton("downloadData", "Download the Predictions⤵️")
    ),
    
    # Show the table with the predictions
    mainPanel(
      DT::dataTableOutput("mytable")
    )
  )
)

# Defining server logic
server <- function(input, output) {
  
  
  reactiveDF<-reactive({
    req(input$file1)
    df <- read.csv(input$file1$datapath, stringsAsFactors = TRUE)
    
    df$predictions <-predict(model, newdata = df)
    return(df)
    
  })
  
  output$mytable = DT::renderDataTable({
    req(input$file1)
    
    return(DT::datatable(reactiveDF(),  options = list(pageLength = 100), filter = c("top")))
  })
  
  
  # Downloadable csv of selected dataset ----
  output$downloadData <- downloadHandler(
    filename = function() {
      paste("data-", Sys.Date(), ".csv", sep="")
    },
    content = function(file) {
      write.csv(reactiveDF(), file, row.names = FALSE)
    }
  )
  
  
  
}

# Run the application
shinyApp(ui = ui, server = server)