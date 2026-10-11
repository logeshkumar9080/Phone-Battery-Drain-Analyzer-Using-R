# ==============================================================
#  Phone Battery Drain Analyzer  -  R Shiny Dashboard
#  Run with:  shiny::runApp()   (from the project root folder)
# ==============================================================
pkgs <- c("shiny", "ggplot2", "dplyr", "DT", "ranger")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
  library(p, character.only = TRUE)
}

GREEN <- "#1B5E3A"; AMBER <- "#F5A623"; LIGHT <- "#7BC47F"
NET_LEVELS <- c("WiFi", "4G", "5G")

# ---------- Data loading + cleaning (same steps as R/01_analysis.R) ----------
load_clean <- function(path = "data/battery_data.csv") {
  d <- read.csv(path, stringsAsFactors = FALSE)
  d <- d[!duplicated(d), ]
  for (col in c("SignalStrength", "BatteryTempC")) {
    d[[col]][is.na(d[[col]])] <- median(d[[col]], na.rm = TRUE)
  }
  d$NetworkType <- factor(d$NetworkType, levels = NET_LEVELS)
  d
}
battery <- load_clean()

# ---------- Train models once when the app starts ----------
set.seed(42)
idx   <- sample(seq_len(nrow(battery)), size = floor(0.8 * nrow(battery)))
train <- battery[idx, ]
test  <- battery[-idx, ]

lm_model <- lm(DrainPerHour ~ ., data = train)
rf_model <- ranger(DrainPerHour ~ ., data = train, num.trees = 200,
                   importance = "impurity", seed = 42)
lm_pred  <- predict(lm_model, newdata = test)
rf_pred  <- predict(rf_model, data = test)$predictions

calc_metrics <- function(actual, pred) {
  c(RMSE = sqrt(mean((actual - pred)^2)),
    MAE  = mean(abs(actual - pred)),
    R2   = 1 - sum((actual - pred)^2) / sum((actual - mean(actual))^2))
}
perf <- data.frame(Model = c("Linear Regression", "Random Forest"),
                   rbind(calc_metrics(test$DrainPerHour, lm_pred),
                         calc_metrics(test$DrainPerHour, rf_pred)))
perf[, -1] <- round(perf[, -1], 3)
best_model <- if (perf$RMSE[1] <= perf$RMSE[2]) "lm" else "rf"
best_label <- if (best_model == "lm") "Linear Regression" else "Random Forest"

cuts <- quantile(train$DrainPerHour, c(0.33, 0.67))   # Low / Normal / High limits
categorise <- function(x) {
  if (x < cuts[1]) "Low" else if (x < cuts[2]) "Normal" else "High"
}

# One prediction function used by the Predict tab
predict_drain <- function(model, nd) {
  if (model == "lm") unname(predict(lm_model, nd))
  else predict(rf_model, data = nd)$predictions
}

num_cols <- setdiff(names(battery)[sapply(battery, is.numeric)], "DrainPerHour")

# ---------- UI ----------
vbox <- function(title, value, sub = NULL) {
  div(class = "vbox", div(class = "vt", title), div(class = "vv", value),
      if (!is.null(sub)) div(class = "vs", sub))
}

ui <- navbarPage(
  title = "Battery Drain Analyzer",
  header = tags$head(tags$style(HTML(sprintf("
    .navbar-default { background-color:%s; border:none; }
    .navbar-default .navbar-brand, .navbar-default .navbar-nav>li>a { color:white !important; }
    .navbar-default .navbar-nav>.active>a { background-color:%s !important; color:#222 !important; }
    .vbox { background:#f1f8f3; border-radius:10px; padding:14px 18px; margin-bottom:15px; }
    .vt { font-size:13px; color:#555; } .vv { font-size:30px; font-weight:700; color:%s; }
    .vs { font-size:12px; color:#888; }
    .result { background:%s; color:white; border-radius:12px; padding:20px; text-align:center; }
    .result .big { font-size:46px; font-weight:700; }
  ", GREEN, AMBER, GREEN, GREEN)))),

  # ---- Tab 1 : Overview ----
  tabPanel("Overview",
    fluidRow(
      column(3, uiOutput("vb_rows")), column(3, uiOutput("vb_mean")),
      column(3, uiOutput("vb_high")), column(3, uiOutput("vb_life"))),
    fluidRow(
      column(7, plotOutput("hist_plot", height = "330px")),
      column(5, plotOutput("cat_plot",  height = "330px"))),
    p("Drain per hour = average percentage of battery lost per hour of active phone use.",
      style = "color:#666; margin-top:10px;")
  ),

  # ---- Tab 2 : Explore ----
  tabPanel("Explore",
    sidebarLayout(
      sidebarPanel(width = 3,
        selectInput("xvar", "Choose a factor:", choices = num_cols,
                    selected = "ScreenBrightness"),
        sliderInput("drain_rng", "Filter drain range (%/hr):",
                    min = floor(min(battery$DrainPerHour)),
                    max = ceiling(max(battery$DrainPerHour)),
                    value = c(floor(min(battery$DrainPerHour)),
                              ceiling(max(battery$DrainPerHour)))),
        checkboxGroupInput("nets", "Network type:", choices = NET_LEVELS,
                           selected = NET_LEVELS)),
      mainPanel(width = 9,
        plotOutput("explore_plot", height = "380px"),
        verbatimTextOutput("explore_cor"))
    )
  ),

  # ---- Tab 3 : Drain factors ----
  tabPanel("Drain Factors",
    fluidRow(
      column(6, plotOutput("cor_plot", height = "380px")),
      column(6, plotOutput("imp_plot", height = "380px"))),
    fluidRow(column(12, plotOutput("net_plot", height = "300px")))
  ),

  # ---- Tab 4 : Model performance ----
  tabPanel("Model Performance",
    fluidRow(column(12, h4("Test-set results (20% unseen data)"),
                    tableOutput("perf_tbl"),
                    p(textOutput("best_txt")))),
    fluidRow(
      column(6, plotOutput("avp_lm", height = "340px")),
      column(6, plotOutput("avp_rf", height = "340px")))
  ),

  # ---- Tab 5 : Predict ----
  tabPanel("Predict My Drain",
    sidebarLayout(
      sidebarPanel(width = 4,
        sliderInput("brightness", "Screen brightness (%)", 5, 100, 60),
        sliderInput("screen_on", "Screen-on time (hours/day)", 0.5, 14, 4, step = 0.5),
        sliderInput("apps", "Apps running in background", 1, 30, 8),
        radioButtons("net", "Network", choices = NET_LEVELS, inline = TRUE),
        sliderInput("signal", "Signal strength (dBm, closer to -50 is better)",
                    -115, -50, -85),
        sliderInput("temp", "Battery temperature (C)", 24, 50, 34, step = 0.5),
        sliderInput("gaming", "Gaming hours/day", 0, 8, 0.5, step = 0.5),
        sliderInput("health", "Battery health (%)", 55, 100, 88),
        checkboxInput("gps", "GPS / Location ON", FALSE),
        checkboxInput("bt", "Bluetooth ON", FALSE),
        checkboxInput("sync", "Background sync ON", TRUE),
        checkboxInput("dark", "Dark mode ON", FALSE)),
      mainPanel(width = 8,
        fluidRow(
          column(5, div(class = "result", div("Predicted drain"),
                        div(class = "big", textOutput("pred_val", inline = TRUE)),
                        div("% per hour"))),
          column(4, vbox("Drain level", textOutput("pred_cat"))),
          column(3, vbox("Battery lasts", textOutput("pred_life")))),
        fluidRow(column(12, h4("What if you optimise your settings?"),
                        uiOutput("savings"))),
        fluidRow(column(12, h4("Personal tips"), uiOutput("tips")))
      )
    )
  ),

  # ---- Tab 6 : Data ----
  tabPanel("Data",
    downloadButton("dl", "Download cleaned data (CSV)"),
    br(), br(), DTOutput("tbl")
  ),

  # ---- Tab 7 : About ----
  tabPanel("About",
    h3("Phone Battery Drain Analyzer"),
    p("An R Shiny dashboard that analyses what drains a smartphone battery and predicts",
      "the drain per hour from usage settings using Linear Regression and Random Forest."),
    p("Dataset: simulated smartphone usage records (5,000 rows, 12 features) created for",
      "this academic mini project. Source file: data/battery_data.csv"),
    p("Built with: R, shiny, ggplot2, dplyr, ranger, DT.")
  )
)

# ---------- Server ----------
server <- function(input, output, session) {

  # ----- Overview -----
  output$vb_rows <- renderUI(vbox("Records analysed", format(nrow(battery), big.mark = ",")))
  output$vb_mean <- renderUI(vbox("Average drain",
                                  paste0(round(mean(battery$DrainPerHour), 2), " %/hr")))
  output$vb_high <- renderUI({
    pct <- round(100 * mean(battery$DrainPerHour >= cuts[2]), 1)
    vbox("High-drain users", paste0(pct, "%"))
  })
  output$vb_life <- renderUI(vbox("Typical battery life",
                                  paste0(round(100 / mean(battery$DrainPerHour), 1), " hrs"),
                                  "active use, full charge"))

  output$hist_plot <- renderPlot({
    ggplot(battery, aes(DrainPerHour)) +
      geom_histogram(bins = 40, fill = GREEN, colour = "white") +
      geom_vline(xintercept = mean(battery$DrainPerHour), colour = AMBER, linewidth = 1.2) +
      labs(title = "Distribution of battery drain", x = "Drain (% per hour)", y = "Count") +
      theme_minimal(base_size = 14)
  })
  output$cat_plot <- renderPlot({
    cat_df <- data.frame(Level = factor(sapply(battery$DrainPerHour, categorise),
                                        levels = c("Low", "Normal", "High")))
    ggplot(cat_df, aes(Level, fill = Level)) + geom_bar() +
      scale_fill_manual(values = c(LIGHT, GREEN, AMBER)) +
      labs(title = "Users by drain level", x = NULL, y = "Count") +
      theme_minimal(base_size = 14) + theme(legend.position = "none")
  })

  # ----- Explore -----
  filtered <- reactive({
    d <- battery[battery$DrainPerHour >= input$drain_rng[1] &
                 battery$DrainPerHour <= input$drain_rng[2] &
                 battery$NetworkType %in% input$nets, ]
    validate(need(nrow(d) > 5, "No data for this filter. Widen the range."))
    d
  })
  output$explore_plot <- renderPlot({
    d <- filtered(); x <- input$xvar
    if (length(unique(d[[x]])) <= 2) {
      ggplot(d, aes(x = factor(.data[[x]]), y = DrainPerHour, fill = factor(.data[[x]]))) +
        geom_boxplot() + scale_fill_manual(values = c(LIGHT, AMBER)) +
        labs(x = paste(x, "(0 = off, 1 = on)"), y = "Drain (% per hour)",
             title = paste("Drain vs", x)) +
        theme_minimal(base_size = 14) + theme(legend.position = "none")
    } else {
      ggplot(d, aes(x = .data[[x]], y = DrainPerHour)) +
        geom_point(alpha = 0.25, colour = GREEN) +
        geom_smooth(method = "lm", formula = y ~ x, colour = AMBER, se = FALSE) +
        labs(x = x, y = "Drain (% per hour)", title = paste("Drain vs", x)) +
        theme_minimal(base_size = 14)
    }
  })
  output$explore_cor <- renderPrint({
    d <- filtered()
    cat("Rows shown:", nrow(d), "\n")
    cat("Correlation of", input$xvar, "with drain:",
        round(cor(d[[input$xvar]], d$DrainPerHour), 3), "\n")
  })

  # ----- Drain factors -----
  output$cor_plot <- renderPlot({
    cr <- sapply(battery[num_cols], function(v) cor(v, battery$DrainPerHour))
    df <- data.frame(Feature = names(cr), Corr = as.numeric(cr))
    ggplot(df, aes(reorder(Feature, Corr), Corr, fill = Corr > 0)) +
      geom_col() + coord_flip() +
      scale_fill_manual(values = c(LIGHT, AMBER)) +
      labs(title = "Correlation with drain", x = NULL, y = "Correlation") +
      theme_minimal(base_size = 13) + theme(legend.position = "none")
  })
  output$imp_plot <- renderPlot({
    imp <- sort(rf_model$variable.importance, decreasing = TRUE)
    df  <- data.frame(Feature = names(imp), Importance = as.numeric(imp))[1:10, ]
    ggplot(df, aes(reorder(Feature, Importance), Importance)) +
      geom_col(fill = GREEN) + coord_flip() +
      labs(title = "Random Forest feature importance (top 10)", x = NULL) +
      theme_minimal(base_size = 13)
  })
  output$net_plot <- renderPlot({
    ggplot(battery, aes(NetworkType, DrainPerHour, fill = NetworkType)) +
      geom_boxplot() + scale_fill_manual(values = c(LIGHT, GREEN, AMBER)) +
      labs(title = "Drain by network type", x = NULL, y = "Drain (% per hour)") +
      theme_minimal(base_size = 13) + theme(legend.position = "none")
  })

  # ----- Model performance -----
  output$perf_tbl <- renderTable(perf, digits = 3)
  output$best_txt <- renderText(paste("Best model on this test set (lowest RMSE):", best_label))
  avp_plot <- function(pred, col, ttl) {
    ggplot(data.frame(Actual = test$DrainPerHour, Predicted = pred), aes(Actual, Predicted)) +
      geom_point(alpha = 0.35, colour = col) +
      geom_abline(slope = 1, intercept = 0, colour = "black", linetype = "dashed") +
      labs(title = ttl) + theme_minimal(base_size = 13)
  }
  output$avp_lm <- renderPlot(avp_plot(lm_pred, GREEN, "Linear Regression: predicted vs actual"))
  output$avp_rf <- renderPlot(avp_plot(rf_pred, AMBER, "Random Forest: predicted vs actual"))

  # ----- Predict -----
  make_nd <- function(brightness, apps, gps, bt, sync, dark) {
    data.frame(ScreenBrightness = brightness, ScreenOnHours = input$screen_on,
               AppsRunning = apps, NetworkType = factor(input$net, levels = NET_LEVELS),
               SignalStrength = input$signal, BatteryTempC = input$temp,
               GPS_On = as.integer(gps), Bluetooth_On = as.integer(bt),
               BackgroundSync_On = as.integer(sync), DarkMode_On = as.integer(dark),
               GamingHours = input$gaming, BatteryHealthPct = input$health)
  }
  current <- reactive({
    predict_drain(best_model,
                  make_nd(input$brightness, input$apps, input$gps, input$bt,
                          input$sync, input$dark))
  })
  optimised <- reactive({
    predict_drain(best_model,
                  make_nd(min(input$brightness, 40), max(3, input$apps - 4),
                          FALSE, FALSE, FALSE, TRUE))
  })

  output$pred_val  <- renderText(sprintf("%.2f", current()))
  output$pred_cat  <- renderText(categorise(current()))
  output$pred_life <- renderText(sprintf("%.1f hrs", 100 / current()))

  output$savings <- renderUI({
    now <- current(); opt <- optimised()
    if (opt >= now - 0.05) {
      p("Your settings are already close to optimal. Nice!")
    } else {
      p(sprintf("Lower brightness (max 40%%), GPS/Bluetooth/background sync off, dark mode on and 4 fewer apps would cut drain from %.2f to %.2f %%/hr.",
                now, opt),
        sprintf("Battery life would go from %.1f to %.1f hours (+%.1f hrs).",
                100 / now, 100 / opt, 100 / opt - 100 / now))
    }
  })

  output$tips <- renderUI({
    t <- list()
    if (input$brightness > 70) t <- c(t, "Brightness is high. Use auto-brightness or keep it below 50%.")
    if (input$gps)  t <- c(t, "Turn GPS off, or set location to 'while using the app'.")
    if (input$sync) t <- c(t, "Switch off background sync for apps you rarely open.")
    if (!input$dark) t <- c(t, "Enable dark mode (saves power on OLED/AMOLED screens).")
    if (input$apps > 12) t <- c(t, "Many apps are running. Close unused apps.")
    if (input$net == "5G") t <- c(t, "5G uses more power. Switch to 4G or WiFi when speed is not needed.")
    if (input$signal < -100) t <- c(t, "Weak signal makes the phone work harder. Use WiFi or move to better coverage.")
    if (input$temp > 40) t <- c(t, "Phone is hot. Remove the case and avoid charging while gaming.")
    if (input$gaming > 3) t <- c(t, "Long gaming sessions are a major drain. Lower graphics or take breaks.")
    if (input$health < 75) t <- c(t, "Battery health is low. Consider a battery replacement.")
    if (length(t) == 0) t <- list("Great! No major battery-draining habits found.")
    tags$ul(lapply(t, tags$li))
  })

  # ----- Data -----
  output$tbl <- renderDT(datatable(battery, options = list(pageLength = 10, scrollX = TRUE)))
  output$dl  <- downloadHandler(
    filename = function() "battery_data_cleaned.csv",
    content  = function(file) write.csv(battery, file, row.names = FALSE))
}

shinyApp(ui, server)
